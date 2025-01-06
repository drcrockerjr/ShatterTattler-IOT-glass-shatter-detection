import os
import threading
from enum import Enum
from train import ModelTrainer

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
        self.upper_freq = 13000
        self.lower_freq = 5000

        self.model_trainer = ModelTrainer()
        self.retrain = retrain


    def initialize(self):


        if self.retrain: 
            self.model_trainer.train_model()



    def run_hub(self):

        self.initialize()

        while not self.shutdown_event:
            pass
