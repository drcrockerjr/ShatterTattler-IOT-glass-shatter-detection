import os
from enum import Enum
from train import ModelTrainer
from predictor import Predictor
import logging
import asyncio
import struct
import time
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from notification import AlertCode, notify_user
from dataclasses import dataclass, field
from contextlib import suppress

from edge_node import BLEEdgeClient

from bleak import BleakClient, BleakScanner

# from hub_ble import discover_edge_devices, run_ble_client#, run_queue_consumer
from dataset import wav_to_feature, index_to_label

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


logger = logging.getLogger(__name__)

esp_name = "XIAOESP32S3_BLE_SERVER"

class Privilege(Enum):
    admin = 1
    user = 2 

class User: 
    def __init__(self, name:str, phone_number:str, privileges: Privilege):
        self.name = name

class Hub():

    def __init__(self, load_trainer:bool= True, retrain:bool=False):
        self.edge_nodes = []
        self.alarm_state = {
            "green": False,
            "yellow": False,
            "red": False
        }

        self.esp_uuids = [
            "00002A57-0000-1000-8000-00805F9B34FB"
        ]

        self.shutdown_event = asyncio.Event()

        self.n_audio_packets = 0

        self.connected_edge_nodes = []

        # Signal Processing
        self.sample_rate = 16000
        self.upper_freq = 6000
        self.lower_freq = 2000

        self.retrain = retrain
        if load_trainer:
            self.model_trainer = ModelTrainer()
        else: 
            self.model_trainer = None
            self.retrain = False
        
        self.predictor = Predictor()

        # Edge Device Config
        self.edge_sample_rate = 16000
        self.num_audio_sec = 3
        self.max_edge_devices = 2
        self.ble_queue = asyncio.Queue()
        self.ble_client_tasks = {}
        self.ble_heartbeat_tasks = {}
        self.audio_buffer = []

        self.client_recv_data = {}

        self.logger = logging.getLogger(__name__)


    def initialize(self):

        if self.retrain: 
            self.logger.info(f" Training model at path: {self.model_trainer.state_path}")
            self.model_trainer.train_model()
            
        self.logger.info(f" Loading state of existing saved model at path: {self.predictor.state_path}")
        self.predictor.load_state()

    async def consume_ble_packets(self):
        while not self.shutdown_event.is_set():
            try:
                epoch, sender, data = await asyncio.wait_for(self.ble_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            await self.process_packet(epoch, sender, data)

    async def process_packet(self, epoch, sender, data):
        if data is None:
            self.logger.debug("No audio data")
            return

        pcm = struct.unpack('<' + 'h' * (len(data)//2), data)
        buffer = self.client_recv_data.setdefault(sender, [])
        buffer.extend(pcm)

        if len(buffer) >= self.edge_sample_rate * self.num_audio_sec:
            # move to a worker
            buf = buffer.copy()
            self.client_recv_data[sender] = []
            asyncio.create_task(self.classify_packet(sender, buf))

        
    async def manage_audio_buffer(self):
        self.logger.info("Starting audio consumer")
        consumer = asyncio.create_task(self.consume_ble_packets())
        await self.shutdown_event.wait()
        consumer.cancel()
        with suppress(asyncio.CancelledError):
            await consumer

    def is_ready_classify(self, data):
        
        # if len(data) > self.edge_sample_rate*self.num_audio_sec:
        if len(data) > self.edge_sample_rate*self.num_audio_sec:
            print(f"{len(data)} -> Whole sample recv")
            return True
        return False


    def classify_packet(self, sender, packet):
        
        uuid = "0440"
        with open(f"audio_wav{self.n_audio_packets}.txt", "a") as f:
            f.write(f"{packet}\n\n\n")

        self.client_recv_data[str(sender)] = [] # only 

        wf = torch.tensor(packet, dtype=torch.float32)
        self.logger.info(f"\n wf shape: {wf.shape}")


        # feature = wav_to_feature(wf=wf, sample_rate=self.edge_sample_rate, new_sample_rate=16000)
        # feature = feature.to(self.model_trainer.device)
        # print(f"device set to{self.model_trainer.device}")
        # print(f"Feature size: {feature.shape}")
                
        feature = wav_to_feature(wf=wf, 
                                    sample_rate=self.edge_sample_rate, 
                                    new_sample_rate=16000,
                                    )

        feature = feature.to(self.predictor.device)

        size_bytes = feature.element_size() * feature.numel()

        prediction = self.predictor.predict([feature])
        self.logger.info(f" Prediction: {prediction.item()} for Audio packet of device: {str(sender)}")

        flag = False

        if prediction.item() == "glassbreak":
            flag = True
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.logger.info(f"Glass break happended from device: {uuid}, Flag: {flag}, Timestamp: {timestamp_str}\n\n")

            # notify_user(AlertCode.GLASS_BREAK, "4pm", "0440")
            flag = False

            plt.plot(packet)
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            plt.title(f"Audio Packet {self.n_audio_packets} Plot pre upscale")
            plt.show(block=True)
        else:
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # f.write(f"Sound dected but NO glass break happended from device: {uuid}, Flag: {flag}, Timestamp: {timestamp_str}\n\n")

            self.logger.info(f"Sound dected but NO glass break happended from device: {uuid}, Flag: {flag}, Timestamp: {timestamp_str}\n\n")

            #notify_user(AlertCode.NO_GLASS_BREAK, "4pm", "0440")


            
            # size_bytes = feature.element_size() * feature.numel()

    async def discover_edge_devices(self):

        devices = []

        try:
            discover_dict = await BleakScanner.discover(adapter="hci1", return_adv=True)
            # dev_dict[0]: BLEDevice, dev_dict[1]: AdvertisementData
            for dev_addr, dev_dict in discover_dict.items():
                # logger.info(f"U: {dev_addr}, N: {dev_dict[0].name}, rssi: {dev_dict[0].rssi}")
                
                if dev_dict[0].name == esp_name:
                    self.logger.info(f"found ESP32 at addr: {dev_addr}, UUIDs: {dev_dict[1]}")

                    if len(devices) < self.max_edge_devices:
                        devices.append(dev_dict[0]) # For Bleak better to connect with BLEDevice class
                        break
                    else:
                        self.logger.info(f"Max Device connection exceeded, not adding device: {dev_addr}")
                else:
                    # logger.info(f"Target device not found. Retrying after 1 second...")
                    # await asyncio.sleep(1)
                    pass

        except Exception as e:
            self.logger.error(f"Encountered error during discovery: {e}")

        return devices



    async def start_edge_ble(self):
        """Continuously scan for new ESP32 devices and spin up clients."""


    async def run_hub(self):
        self.initialize()

        # start the audio‐buffer consumer
        audio_consumer = asyncio.create_task(self.manage_audio_buffer())

        # launch discovery in background
        # discover_task = asyncio.create_task(self.start_edge_ble())

        try:
            while not self.shutdown_event.is_set():
                devices = await self.discover_edge_devices()
                for dev in devices:
                    if dev.address in {c.device.address for c in self.connected_edge_nodes}:
                        continue

                    client = BLEEdgeClient(
                        device=dev,
                        queue=self.ble_queue,
                        esp_uuids=self.esp_uuids,
                        shutdown_event=self.shutdown_event
                    )
                    self.connected_edge_nodes.append(client)

                    # connect + notifications
                    self.ble_client_tasks[dev.address] = asyncio.create_task(client.connect())
                    # heartbeat
                    self.ble_heartbeat_tasks[dev.address] = asyncio.create_task(
                        client.heartbeat_loop(interval=15.0)
                    )

                await asyncio.sleep(2.0)

        except KeyboardInterrupt:
            self.shutdown_event.set()
        finally:
            # cancel audio consumer
            audio_consumer.cancel()
            with suppress(asyncio.CancelledError):
                await audio_consumer

            # cancel heartbeats
            for addr, hb in self.ble_heartbeat_tasks.items():
                hb.cancel()
                with suppress(asyncio.CancelledError):
                    await hb

            # call each client.disconnect() to stop notifies + close BLE link
            for client in self.connected_edge_nodes:
                await client.disconnect()

            # make sure any outstanding connect() calls finish
            await asyncio.gather(*self.ble_client_tasks, return_exceptions=True)

            self.logger.info("Hub cleanly shut down")

if __name__=="__main__":
    hub = Hub(load_trainer=True, retrain=False)

    asyncio.run(hub.run_hub())


