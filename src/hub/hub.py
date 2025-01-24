import os
import threading
from enum import Enum
from train import ModelTrainer
import logging
import asyncio
import struct
import time

from bleak import BleakClient

from hub_ble import discover_edge_devices, run_ble_client, run_queue_consumer

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

        self.shutdown_event = threading.Event()


        # Signal Processing
        self.sample_rate = 16000
        self.upper_freq = 6000
        self.lower_freq = 2000

        self.model_trainer = ModelTrainer()
        self.retrain = retrain

        # Edge Device Config
        self.max_edge_devices = 2
        self.queue = asyncio.Queue()
        self.ble_tasks = []

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

        

    async def start_edge_ble(self):

        devices = await discover_edge_devices()

        if len(devices) > 0:
            # await asyncio.gather((run_ble_client(device, self.queue)for device in devices))

            self.ble_tasks.extend([asyncio.create_task(run_ble_client(device, self.queue)for device in devices)])
            self.ble_tasks.append(asyncio.create_task(run_queue_consumer(self.queue)))
        else:
            self.logger.info(f"No edge devices to connect to")

        # try:
        #     await asyncio.gather(*self.ble_tasks, consumer_task)
        # except Exception as e:
        #     self.logger.error(f"An error occurred: {e}")
        

    async def run_queue_consumer(self, queue: asyncio.Queue):
        logger.info("Starting queue consumer")
    
        init_recv_t = 0
        while True:
            # Use await asyncio.wait_for(queue.get(), timeout=1.0) if you want a timeout for getting data.
            epoch, sender, data = await queue.get()
            if data is None:
                logger.info(
                    "Got message from client about disconnection. Exiting consumer loop..."
                )
                break
            else:

                
                # Byte Array sent from ESP32
                data = struct.unpack('<' + 'H' *(len(data) // 2), data)

                logger.info("Received callback data via async queue at %s: %r", epoch, data)

                if self.client_recv_data[sender] is None:
                    self.client_recv_data[sender] = []
                    self.client_recv_data[sender].extent(data)

                with open(f"{sender}_data.txt", "a") as f:
                    f.write(self.client_recv_data[sender] + "\n\n")
                
                # if 1 in data:
                #     init_recv_t = time.time()
            
                # if 1000 in data:
                #     logger.info(f"Recv 1000 Samples after {time.time() - init_recv_t} s")


    def run_hub(self):

        self.initialize()

        asyncio.run(self.start_edge_ble())

        # while not self.shutdown_event:
        #     pass


if __name__=="__main__":
    hub = Hub(retrain=False)

    hub.run_hub()


