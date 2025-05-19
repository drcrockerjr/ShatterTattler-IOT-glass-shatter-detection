import os
from enum import Enum
from train import ModelTrainer
import logging
import asyncio
import struct
import time
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from notification import AlertCode, notify_user

from bleak import BleakClient

from hub_ble import discover_edge_devices, run_ble_client#, run_queue_consumer
from dataset import wav_to_feature, index_to_label

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Privilege(Enum):
    admin = 1
    user = 2  # Define different privilege levels for users

class User:
    def __init__(self, name: str, phone_number: str, privileges: Privilege):
        self.name = name
        self.phone_number = phone_number  # Store user's contact info
        self.privileges = privileges      # Store user's access level

class Hub:
    def __init__(self, load_trainer: bool = True, retrain: bool = False):
        # List of connected edge devices
        self.edge_nodes = []
        # Track alarm states for visual indicators
        self.alarm_state = {"green": False, "yellow": False, "red": False}

        # Event used to signal shutdown across coroutines
        self.shutdown_event = asyncio.Event()

        # Count how many audio packets have been processed
        self.n_audio_packets = 0

        # Audio processing parameters
        self.sample_rate = 16000
        self.lower_freq = 2000
        self.upper_freq = 6000

        # Trainer setup: optionally load or disable retraining
        self.retrain = retrain
        if load_trainer:
            self.model_trainer = ModelTrainer()
        else:
            self.model_trainer = None
            self.retrain = False

        # BLE edge device configuration
        self.edge_sample_rate = 16000
        self.max_edge_devices = 2
        self.queue = asyncio.Queue()   # Queue for receiving BLE audio data
        self.ble_tasks = []            # Tasks handling BLE clients
        self.audio_buffer = []         # Buffer to store raw audio packets

        # Per-client accumulated data
        self.client_recv_data = {}

        # Use a class-specific logger
        self.logger = logging.getLogger(__name__)

    def initialize(self):
        """
        If retraining is enabled, train the model from scratch.
        Otherwise, skip (or load existing model state if implemented).
        """
        if self.retrain:
            self.logger.info(f"Training model at path: {self.model_trainer.state_path}")
            self.model_trainer.train_model()
        else:
            # Placeholder for loading a pre-trained model if desired
            pass

    async def manage_audio_buffer(self, queue: asyncio.Queue):
        """
        Consume audio data from the queue, unpack it, accumulate per client,
        and trigger classification once enough samples have arrived.
        """
        self.logger.info("Starting queue consumer")
        init_recv_t = time.time()
        byte_cnt = 0

        while not self.shutdown_event.is_set():
            try:
                # Use a timeout to periodically check for shutdown
                epoch, sender, data = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if data is None:
                self.logger.info("No audio data from BLE available")
            else:
                # Update total bytes received
                byte_cnt += len(data)
                # Unpack incoming byte array into unsigned 16-bit samples
                data = struct.unpack('<' + 'H' * (len(data) // 2), data)

                logger.info(
                    "Received %d bytes/sec: %r from %s",
                    byte_cnt / (time.time() - init_recv_t), data, sender
                )

                # Initialize per-sender storage if first packet
                sender_key = str(sender)
                self.client_recv_data.setdefault(sender_key, [])
                # Append new samples
                self.client_recv_data[sender_key].extend(data)
                # Keep a chronological log for debugging
                self.audio_buffer.append((epoch, data))

                # Check each client's buffer for readiness to classify
                for sender_key, samples in self.client_recv_data.items():
                    logger.info(f"\n\nBuffer Len for {sender_key}: {len(samples)}\n\n")
                    if self.is_ready_classify(samples):
                        packet = samples
                        self.n_audio_packets += 1

                        # Dump raw packet to a file for offline inspection
                        with open(f"audio_wav{self.n_audio_packets}.txt", "a") as f:
                            f.write(f"{packet}\n\n\n")
                            # Convert to tensor for feature extraction
                            wf = torch.tensor(packet, dtype=torch.float32)
                            feature = wav_to_feature(
                                wf=wf,
                                sample_rate=self.edge_sample_rate,
                                new_sample_rate=16000
                            )
                            feature = feature.to(self.model_trainer.device)
                            size_bytes = feature.element_size() * feature.numel()

                            # Run the model to get predictions
                            output, _ = self.model_trainer.model(
                                feature, self.model_trainer.model.init_hidden(1)
                            )
                            prediction = torch.max(output, dim=1).indices

                            # Log feature details and prediction
                            f.write(f"Data Shape: {feature.shape}, dtype: {feature.dtype}, Datasize: {size_bytes} bytes\n")
                            f.write(f"Prediction: {prediction}\n")

                        # Reset this client's buffer after classification
                        self.client_recv_data[sender_key] = []

                # Optionally: persist full buffer history for auditing
                if self.audio_buffer:
                    with open("buffer_data.txt", "a") as f:
                        for epoch_ts, packet in self.audio_buffer:
                            f.write(f"Epoch: {epoch_ts}\n")
                            for val in packet:
                                f.write(f"{val}\n")
                            f.write("\n")
                        f.flush()

    def is_ready_classify(self, data):
        """
        Check whether enough samples have been collected to form
        a complete segment for classification.
        """
        if len(data) > self.edge_sample_rate:
            print(f"{len(data)} -> Whole sample recv")
            return True
        return False

    async def start_edge_ble(self):
        """
        Discover BLE edge devices, start connection tasks, and
        launch the consumer for incoming audio data.
        """
        devices = await discover_edge_devices()
        if not devices:
            self.logger.info("No edge devices to connect to")
            return

        # Kick off BLE client tasks for each discovered device
        ble_clients = [run_ble_client(device, self.queue) for device in devices]
        consumer_task = asyncio.create_task(self.manage_audio_buffer(self.queue))

        try:
            await asyncio.gather(*ble_clients, return_exceptions=True)
            await consumer_task
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")

    async def run_hub(self):
        """
        Main entry point to initialize the system and keep
        it running until a shutdown signal is received.
        """
        self.initialize()
        main_ble_task = asyncio.create_task(self.start_edge_ble())

        # Keep the hub alive, checking for shutdown periodically
        while not self.shutdown_event.is_set():
            await asyncio.sleep(1.0)

        await main_ble_task
        self.logger.info("Shutting down Hub")

if __name__ == "__main__":
    # Instantiate without loading trainer to skip model operations
    hub = Hub(load_trainer=False, retrain=False)
    asyncio.run(hub.run_hub())