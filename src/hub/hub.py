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
from notification import AlertCode, notify_user, periodic_report
from dataclasses import dataclass, field
from contextlib import suppress

from edge_node import BLEEdgeClient

from bleak import BleakClient, BleakScanner

# from hub_ble import discover_edge_devices, run_ble_client#, run_queue_consumer
from dataset import wav_to_feature, index_to_label

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


logger = logging.getLogger(__name__)

esp_name = "XIAOESP32S3_BLE_SERVER"
alarm_name = "XIAOESP32S3_BLE_ALARM"
charUUID = "616e1a1b-b7e4-4277-a230-5af28c1201a6" #for the alarm node

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
            #"green": False,
            "yellow": 0,
            "red": 0
        }

        self.esp_uuids = [
            "00002A57-0000-1000-8000-00805F9B34FB", # Edge node uuid
            "616e1a1b-b7e4-4277-a230-5af28c1201a6" # Alarm node uuid
        ]

        self._supported_edge_names = [
            "XIAOESP32S3_BLE_SERVER",
            "XIAOESP32S3_BLE_ALARM"
        ]

        self.shutdown_event = asyncio.Event()

        self.n_audio_packets = 0

        self.connected_edge_nodes = {}

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
        self.alarm_duration = 30 #alarm duration

        self.client_recv_data = {}

        self.logger = logging.getLogger(__name__)


    def initialize(self):

        if self.retrain: 
            self.logger.info(f" Training model at path: {self.model_trainer.state_path}")
            self.model_trainer.train_model()
            
        self.logger.info(f" Loading state of existing saved model at path: {self.predictor.state_path}")
        self.predictor.load_state("trained_model.pt")

    """        
    async def manage_audio_buffer(self):
        self.logger.info("Starting audio consumer")

        while not self.shutdown_event.is_set():
            try:
                epoch, sender, data = await asyncio.wait_for(self.ble_queue.get(), timeout=1.0)
                self.logger.info(f"Recv data from sender: {sender}, at epoch: {epoch}")
            except asyncio.TimeoutError:
                # self.logger.info(f"Timeout occured before Recv data in mange buffer")
                continue
            self.n_audio_packets += 1
            classify = asyncio.create_task(self.classify_packet(epoch, sender, data))
            self.logger.info(f"Finished classify")
        with suppress(asyncio.CancelledError):
            await classify
    """

    async def manage_audio_buffer(self):
        self.logger.info("Starting audio consumer")
        while not self.shutdown_event.is_set():
            try:
                epoch, sender, data = await asyncio.wait_for(self.ble_queue.get(), timeout=1.0)
                self.logger.info(f"Recv data from {sender} at {epoch}")
            except asyncio.TimeoutError:
                self.logger.debug("Queue empty; looping again")
                continue

            self.n_audio_packets += 1
            # fire-and-forget
            asyncio.create_task(self.classify_packet(epoch, sender, data))
            self.logger.info("Scheduled classify task")

    async def classify_packet(self, epoch, sender, packet):
        
        uuid = "0440"
        with open(f"audio_wav{self.n_audio_packets}.txt", "a") as f:
            f.write(f"{packet}\n\n\n")

        # self.client_recv_data[str(sender)] = [] # only 

        wf = torch.tensor(packet, dtype=torch.float32)
        self.logger.info(f"\n Wf shape: {wf.shape}\n")


        # feature = wav_to_feature(wf=wf, sample_rate=self.edge_sample_rate, new_sample_rate=16000)
        # feature = feature.to(self.model_trainer.device)
        # print(f"device set to{self.model_trainer.device}")
        # print(f"Feature size: {feature.shape}")
                
        feature = wav_to_feature(wf=wf, 
                                    sample_rate=self.edge_sample_rate, 
                                    new_sample_rate=None,
                                    )
        
        self.logger.info(f"\n Produced feature with shape: {feature.shape}\n")

        feature = feature.to(self.predictor.device)

        size_bytes = feature.element_size() * feature.numel()

        prediction = self.predictor.predict(feature)
        self.logger.info(f" Prediction: {prediction} for Audio packet of device: {str(sender)}")

        flag = False

        if prediction == "glassbreak":
        #if True:
            flag = True
            asyncio.create_task(self.send_alarm_cmd())
            self.alarm_state["red"] += 1
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.logger.info(f"Glass break happended from device: {uuid}, Flag: {flag}, Timestamp: {timestamp_str}\n\n")

            # notify_user(AlertCode.GLASS_BREAK, "4pm", "0440")
            flag = False

        else:
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # f.write(f"Sound dected but NO glass break happended from device: {uuid}, Flag: {flag}, Timestamp: {timestamp_str}\n\n")
            self.alarm_state["yellow"] += 1
            self.logger.info(f"Sound dected but NO glass break happended from device: {uuid}, Flag: {flag}, Timestamp: {timestamp_str}\n\n")

            #notify_user(AlertCode.NO_GLASS_BREAK, "4pm", "0440")
            
            # size_bytes = feature.element_size() * feature.numel()
        self.logger.info(f"finished classify function")
        # plt.plot(packet)
        # plt.xlabel("Time")
        # plt.ylabel("Amplitude")
        # plt.title(f"Audio Packet Plot ")
        # plt.show(block=False)

    async def discover_edge_devices(self):

        devices = []

        try:
            discover_dict = await BleakScanner.discover(adapter="hci1", return_adv=True)
            # dev_dict[0]: BLEDevice, dev_dict[1]: AdvertisementData
            for dev_addr, dev_dict in discover_dict.items():
                # logger.info(f"U: {dev_addr}, N: {dev_dict[0].name}, rssi: {dev_dict[0].rssi}")
                
                if dev_dict[0].name in self._supported_edge_names:
                    self.logger.info(f"Found supported ESP32 at addr: {dev_addr}, UUIDs: {dev_dict[1]}")
                    self.logger.info(f"Adding device to devices")
                    devices.append(dev_dict[0]) # For Bleak better to connect with BLEDevice class
                else:
                    # logger.info(f"Target device not found. Retrying after 1 second...")
                    # await asyncio.sleep(1)
                    pass

        except Exception as e:
            self.logger.error(f"Encountered error during discovery: {e}")

        return devices
    
    async def send_alarm_cmd(self):
        
        #loop thru all connected clients
        for client in self.connected_edge_nodes.values():

            #sets alarms for client
            if client.alarm:
                try:
                    #set alarm and clears after 30 seconds
                    await client.set_alarm()
                    self.logger.info(f"Alarm ON")
                    
                    await asyncio.sleep(self.alarm_duration)
                                        
                    await client.clear_alarm()
                    self.logger.info(f"Alarm OFF")

                except Exception as e:
                    self.logger.error(f"Failed to send alarm cmd: {e}")


    async def heartbeat_servicer(
        self,
        client:BLEEdgeClient,
        interval: float = 30.0,
        max_retries: int = 5,
        retry_delay: int = 5
    ):
        
        try:
            while True:
                await asyncio.sleep(interval)
                alive = client.is_connected()
                if not alive:
                    self.logger.warning(f"Heartbeat: {client.addr} disconnected; reconnecting…")
                    for attempt in range(1, max_retries + 1):
                        try:
                            conn_exists = await client.connect()
                            if conn_exists:
                                self.logger.info(f"Reconnected to {client.addr}")
                                break
                            else:
                                self.logger.error(
                                    f"Reconnect attempt {attempt} failed (still disconnected)"
                                )
                        except Exception as e:
                            self.logger.error(f"Reconnect attempt {attempt} error: {e}")

                        if attempt < max_retries:
                            await asyncio.sleep(retry_delay)
                        else:
                            self.logger.error(
                                f"Failed to {client.addr} reconnect after {max_retries} attempts."
                            )
                            
                            # client.disconnect()

                            # self.connected_edge_nodes.pop(client.addr, None) # Get rid of dead edge node

                            # return
                            
                else:
                    self.logger.debug(f"Heartbeat OK: {client.addr}")
        except asyncio.CancelledError:
            self.logger.info("Heartbeat task cancelled")
            raise

    async def notification_servicer(self, 
                                    minute_interval: int = 5):
        pass

    #only reads from one edge device
    async def read_device_battery(self):
        for nodes in self.edge_nodes:
                if self.edge_nodes.is_alarm() == 1:
                    device_vbat = self.edge_nodes.get_vbat()
                else:
                    device_vbat = "failed to read battery"

    async def report_servicer(
        self,
        report_interval: float = 60.0,     # seconds between reports
        device_id: str = "ALL_DEVICES"
    ):
        """
        Every report_interval seconds:
          1) grab the current time
          2) call your periodic_report logic
          3) reset counts / move prev_time_stamp forward
        """
        prev_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        while not self.shutdown_event.is_set():
            await asyncio.sleep(report_interval)

            curr_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            #read battery voltage from client :
            device_vbat = await self.read_battery_level() #only reports battery from one edge device

            # call out to your free function (or you can inline it here)
            periodic_report(
                alarm_state=self.alarm_state,
                curr_time_stamp=curr_ts,
                prev_time_stamp=prev_ts,
                device_id=device_id,
                device_vbat=device_vbat,
                report_interval=report_interval
            )

            # reset your counters if you want "since last report"
            self.alarm_state["red"] = 0
            self.alarm_state["yellow"] = 0
            prev_ts = curr_ts

    async def run_hub(self):
        self.initialize()

        # start the audio‐buffer consumer
        audio_consumer = asyncio.create_task(self.manage_audio_buffer())

        report_task = asyncio.create_task(self.report_servicer(
            report_interval= 60*5  # every 5 minutes
        ))

        # launch discovery in background
        # discover_task = asyncio.create_task(self.start_edge_ble())

        try:
            while not self.shutdown_event.is_set():
                if len(self.connected_edge_nodes.values()) < self.max_edge_devices: # Only discover if nodes
                    devices = await self.discover_edge_devices()
                    for dev in devices:
                        if dev.address in self.connected_edge_nodes.keys():
                            continue

                        client = BLEEdgeClient( # TODO: make client initialize with sample rate and number of seconds of audio
                            device=dev,
                            queue=self.ble_queue,
                            esp_uuids= self.esp_uuids,
                            samples_per_window=self.edge_sample_rate * self.num_audio_sec,
                            shutdown_event=self.shutdown_event, 
                            alarm = True if dev.name == alarm_name else False
                        )
                        self.connected_edge_nodes[dev.address] = client

                        self.logger.info(f"Hub starting connct task to client at addr: {dev.address}")

                        # connect + notifications
                        self.ble_client_tasks[dev.address] = asyncio.create_task(client.connect())

                        self.logger.info(f"Hub starting heartbeat task to client at addr: {dev.address}")

                        # heartbeat
                        self.ble_heartbeat_tasks[dev.address] = asyncio.create_task(
                            self.heartbeat_servicer(client=client, interval=15.0)
                        )
                await asyncio.sleep(1.0)

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
            for client in self.connected_edge_nodes.values():
                if not client.alarm:
                    await client.disconnect()

            # make sure any outstanding connect() calls finish
            await asyncio.gather(*self.ble_client_tasks, return_exceptions=True)

            self.logger.info("Hub cleanly shut down")

if __name__=="__main__":
    hub = Hub(load_trainer=False, retrain=False)

    asyncio.run(hub.run_hub())


