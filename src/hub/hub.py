import os
import threading
from enum import Enum
from train import ModelTrainer
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class Privilege(Enum):
    admin = 1
    user = 2 

class User: 
    def __init__(self, name:str, phone_number:str, privileges: Privilege):
        self.name = name

class Hub():

    def __init__(self, retrain:bool=True):
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

        self.logger = logging.getLogger(__name__)


    def initialize(self):

        if self.retrain: 
            self.logger.info(f"Training model at path: {self.model_trainer.model_path}")
            self.model_trainer.train_model()
        else: 
            self.model_trainer.load_state()

        




    def run_hub(self):

        self.initialize()

        # while not self.shutdown_event:
        #     pass


if __name__=="__main__":
    hub = Hub(retrain=True)

    hub.run_hub()


