import os
import threading
from enum import Enum

class Privilege(Enum):
    admin = 1
    user = 2 

class User: 
    def __init__(self, name:str, phone_number:str, privileges: Privilege):
        self.name = name

class Hub():

    def __init__(self):
        self.clients = []
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





    def run_hub(self):
        while not self.shutdown_event:
            pass
