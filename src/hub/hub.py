import os
from enum import Enum
from train import ModelTrainer
import logging
import asyncio
import struct
import time

from bleak import BleakClient

from hub_ble import discover_edge_devices, run_ble_client#, run_queue_consumer

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


logger = logging.getLogger(__name__)

class Privilege(Enum):
    admin = 1
    user = 2 

class User: 
    def __init__(self, name:str, phone_number:str, privileges: Privilege):
        self.name = name

class Hub():

    def __init__(self, retrain:bool=False):
        self.edge_nodes = []
        self.alarm_state = {
            "green": False,
            "yellow": False,
            "red": False
        }

        self.shutdown_event = asyncio.Event()


        # Signal Processing
        self.sample_rate = 16000
        self.upper_freq = 6000
        self.lower_freq = 2000

        # self.model_trainer = ModelTrainer()
        self.retrain = retrain

        # Edge Device Config
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

        init_recv_t = 0
        byte_cnt = 0

        client_recv_data = {}
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


                if str(sender) not in client_recv_data:
                    client_recv_data[str(sender)] = []
                client_recv_data[str(sender)].extend(data)

                self.audio_buffer.append((epoch, data))
                
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

    def is_new_measurement_ready(self):
        
        for sender in self.client_recv_data.keys():

            if len(self.client_recv_data[sender]) >= self.sample_rate:

                return True

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
    hub = Hub(retrain=False)

    asyncio.run(hub.run_hub())


