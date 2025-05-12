import os
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from tqdm import tqdm
from datetime import datetime
from model import AudioLSTM
import logging
from torch.utils.tensorboard import SummaryWriter
from dataset import index_to_label

from typing import Any, Dict, Optional



# def collate_fn(batch):

#     # A data tuple has the form:
#     # waveform, sample_rate, label, speaker_id, utterance_number

#     tensors, targets = [], []

#     # Gather in lists, and encode labels as indices
#     for feature, label in batch:
#         tensors += [feature]
#         targets += [label_to_index(label)]

#     # Group the list of tensors into a batched tensor
#     tensors = pad_sequence(tensors)
#     targets = torch.stack(targets)

#     return tensors, targets
# def collate_fn(batch):

#     print("called collate function")
#     features, labels = zip(*batch)  # Unzip the batch into features and labels

#     # Pad sequences to the maximum length in the batch
#     features = pad_sequence(features, batch_first=True, padding_value=0.0)
#     labels = torch.tensor([label_to_index(label) for label in labels])

#     return features, labels


class Predictor():
    def __init__(self, 
                 only_glass: bool = True):
    
        self.device = 'cpu'
        self.batch_size = 128

        self.logger = logging.getLogger(__name__)

        if torch.cuda.is_available():
            self.device = 'cuda'
            num_workers = 0
            pin_memory = True
        

        self.model = AudioLSTM(n_feature=168, out_feature=3)
        self.model.to(self.device)
        print(self.model)

        mdl_dir = "saved_model"
        # os.makedirs(mdl_dir, exist_ok=True)

        self.state_path = os.path.join(mdl_dir, "model.pt")

        # lr = 0.01
        # weight_decay = 0.0001

        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # self.criterion = torch.nn.CrossEntropyLoss()


        # log_dir = 'logs/' + datetime.now().strftime('%B%d_%H_%M_%S')
        # self.writer = SummaryWriter(log_dir)
        
    def log_scalars(self, global_tag, metric_dict, global_step):

        for tag, value in metric_dict.items():
            self.writer.add_scalar(f"{global_tag}/{tag}", value, global_step)
            
                
    def predict(self, feature_list):


        output, _ = self.model(feature_list, self.model.init_hidden(len()))

        prediction = torch.max(output, dim=1).indices
        # print(f"[{idx}]Preiction: {prediction}, Shape: {prediction.shape}\n")

        self.logger(f"Prediction: {index_to_label(prediction.item())} \n\n")

        # if index_to_label(prediction.item()) == "glassbreak":
        #     flag = True
        # if flag == True:
        #     timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #     f.write(f"Glass break happended from device: {uuid}, Flag: {flag}, Timestamp: {timestamp_str}\n\n")

        #     notify_user(AlertCode.GLASS_BREAK, "4pm", "0440")
        #     flag = False

        return index_to_label(prediction.item())

    def save_state(self):

        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }
            , self.state_path)

    def load_state(self, path: Optional[str] = None):
        
        if path is None:
            dict = torch.load(self.state_path)
        else:
            dict = torch.load(path)
            

        self.epoch = dict["epoch"]
        self.model.load_state_dict(dict["model_state_dict"])
        # self.optimizer.load_state_dict(dict["optimizer_state_dict"])
