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

        self.shutdown_event = asyncio.Event()

        self.n_audio_packets = 0
        


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

        # Edge Device Config
        self.edge_sample_rate = 16000
        self.audio_seconds = 3
        self.max_edge_devices = 2
        self.queue = asyncio.Queue()
        self.ble_tasks = []
        self.audio_buffer = []

        self.client_recv_data = {}

        self.logger = logging.getLogger(__name__)


    def initialize(self):

        if self.retrain: 
            self.logger.info(f" Training model at path: {self.model_trainer.state_path}")
            self.model_trainer.train_model()
        else: 
            # self.logger.info(f" Loading state of existing saved model at path: {self.model_trainer.state_path}")
            # self.model_trainer.load_state()
            pass

        
    async def manage_audio_buffer(self, queue: asyncio.Queue):
        self.logger.info("Starting queue consumer")

        init_recv_t = time.time()
        byte_cnt = 0

        while not self.shutdown_event.is_set():
            # Use await asyncio.wait_for(queue.get(), timeout=1.0) if you want a timeout for getting data.
            try:
                epoch, sender, data = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if data is None:
                self.logger.info(
                    "No audio data from BLE available"
                )
                
            else:
                # Add count of bytes recv to byte counter

                byte_cnt += len(data)
                # Byte Array sent from ESP32
                data = struct.unpack('<' + 'H' *(len(data) // 2), data)

                logger.info("Received callback data via async queue at %s: %r, from %s", (byte_cnt / (time.time() - init_recv_t)), data, sender)


                if str(sender) not in self.client_recv_data:
                    self.client_recv_data[str(sender)] = []
                self.client_recv_data[str(sender)].extend(data)

                self.audio_buffer.append((epoch, data))

                for sender in self.client_recv_data.keys():

                    logger.info(f"\n\nBuffer Len: {len(self.client_recv_data[str(sender)])}\n\n")
                    if self.is_ready_classify(self.client_recv_data[str(sender)]) == True:
                        
                        packet = self.client_recv_data[str(sender)]
                        self.n_audio_packets += 1

                        uuid = "0440"
                        with open(f"audio_wav{self.n_audio_packets}.txt", "a") as f:
                            f.write(f"{packet}\n\n\n")

                            self.client_recv_data[str(sender)] = []

                            # plt.plot(packet)
                            # plt.xlabel("Time")
                            # plt.ylabel("Amplitude")
                            # plt.title(f"Audio Packet {self.n_audio_packets} Plot pre upscale")
                            # plt.show(block=True)
                            
                            wf = torch.tensor(packet, dtype=torch.float32)
                        
                            feature = wav_to_feature(wf=wf, sample_rate=self.edge_sample_rate, new_sample_rate=16000)

                            feature = feature.to(self.model_trainer.device)

                            size_bytes = feature.element_size() * feature.numel()

                            output, _ = self.model_trainer.model(feature, self.model_trainer.model.init_hidden(1))

                            prediction = torch.max(output, dim=1).indices

                            # prediction = "gunshot"

                            # feature = torch.tensor(packet, dtype=torch.float32)
                            
                            size_bytes = feature.element_size() * feature.numel()
                            f.write(f"\n\n Data Shape: {feature.shape}, dtype: {feature.dtype}, Datasize: {size_bytes}\n")
                            # f.write(f"\n\n Prediction: {index_to_label(prediction.item())}\n")

                            f.write(f"\n\n Prediction: {prediction}\n")
                            flag = False 
                            if prediction == "glassbreak":
                            # if index_to_label(prediction.item()) == "glassbreak":i
                                flag = True
                            
                            """ 
                            if self.n_audio_packets == 8:
                                flag = True
                            if flag == True:
                                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                f.write(f"Glass break happended from device: {uuid}, Flag: {flag}, Timestamp: {timestamp_str}\n\n")

                                self.logger.info(f"Glass break happended from device: {uuid}, Flag: {flag}, Timestamp: {timestamp_str}\n\n")

                                # notify_user(AlertCode.GLASS_BREAK, "4pm", "0440")
                                flag = False
                            else:
                                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                # f.write(f"Sound dected but NO glass break happended from device: {uuid}, Flag: {flag}, Timestamp: {timestamp_str}\n\n")

                                self.logger.info(f"Sound dected but NO glass break happended from device: {uuid}, Flag: {flag}, Timestamp: {timestamp_str}\n\n")

                                #notify_user(AlertCode.NO_GLASS_BREAK, "4pm", "0440")
                                
                            
                            if self.n_audio_packets == 8:
                                logger.info(f"\n\n\nrecv 8 audio packets after {time.time() - init_recv_t}\n\n\n")

                                plt.plot(packet)
                                plt.xlabel("Time")
                                plt.ylabel("Amplitude")
                                plt.title(f"Audio Packet {self.n_audio_packets} Plot pre upscale")
                                plt.show()
                            # if idx == len(eval_loader):
                            #     f.write(f"\n\n Finished after: {time.time() - start_t}")

                            """


                # if self.n_audio_packets == 8:
                #     logger.info(f"\n\n\nrecv 8 audio packets after {time.time() - init_recv_t}\n\n\n")

                #             plt.plot(packet)
                #             plt.xlabel("Time")
                #             plt.ylabel("Amplitude")
                #             plt.title(f"Audio Packet {self.n_audio_packets} Plot pre upscale")
                #             plt.show(block=True)


                if len(self.audio_buffer) >= 1:

                    with open(f"buffer_data.txt", "a") as f:
                        for audio_packet in self.audio_buffer:
                            f.write(f"Epoch: {audio_packet[0]}\n\n")
                            for val in audio_packet[1]:  
                                f.write(str(val) + "\n")
                            f.write("\n")
                            f.flush()

                # if 1 in data:
                #     init_recv_t = time.time()

                # if 1000 in data:
                #     logger.info(f"Recv 1000 Samples after {time.time() - init_recv_t} s")

    def is_ready_classify(self, data):
        
        # self.audio_seconds = 3
        if len(data) > self.edge_sample_rate * self.audio_seconds:
            print(f"{len(data)} -> Whole sample recv")
            return True
        return False


    async def start_edge_ble(self):

        devices = await discover_edge_devices()
        
        ble_device_clients = []
        if len(devices) > 0:
            # await asyncio.gather((run_ble_client(device, self.queue)for device in devices))

            ble_device_clients.extend([run_ble_client(device, self.queue) for device in devices])
            consumer_task = asyncio.create_task(self.manage_audio_buffer(self.queue))
        else:
            self.logger.info(f"No edge devices to connect to")

        try:
            await asyncio.gather(*ble_device_clients, return_exceptions=True)
            await asyncio.wait(consumer_task)
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
        


    async def run_hub(self):

        self.initialize()

        main_ble_task = asyncio.create_task(self.start_edge_ble())

        while not self.shutdown_event.is_set():
            await asyncio.sleep(1.0)

        await main_ble_task

        self.logger.info(f"Shutting Down Hub")
        

if __name__=="__main__":
    hub = Hub(load_trainer=False, retrain=False)

    asyncio.run(hub.run_hub())


