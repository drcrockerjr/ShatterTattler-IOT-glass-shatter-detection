import os
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
from model import AudioLSTM
from dataset import SEDetectionDataset, train_val_dataset
from preprocess import preprocess
from torch.utils.tensorboard import SummaryWriter



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


class ModelTrainer():
    def __init__(self, 
                 data_root: str = f"../../data/VOICe_clean/", 
                 preprocess_data: bool = False):

        data_root = f"../../data/VOICe_clean/"
        assert os.path.exists(data_root), f"VOICe Dataset path doesnt exist"

        csv_file="audio_info.csv"

        if preprocess_data:
            preprocess(num_synth=80, data_root=data_root) 


        dataset = SEDetectionDataset(csv_file, data_root)

        self.datasets = train_val_dataset(dataset=dataset)

        print("Train set size: " + str(len(self.datasets['train'])))
        print("Test set size: " + str(len(self.datasets['val'])))

        self.device = 'cpu'
        num_workers = 0
        pin_memory = False
        self.batch_size = 128

        if torch.cuda.is_available():
            self.device = 'cuda'
            num_workers = 2
            pin_memory = True

        self.train_loader = torch.utils.data.DataLoader(
            self.datasets['train'], 
            num_workers=num_workers if self.device is 'gpu' else 0,
            batch_size=self.batch_size, 
            # collate_fn=collate_fn,
            shuffle=True, 
            drop_last=True,
            pin_memory=pin_memory)
        
        self.eval_loader = torch.utils.data.DataLoader(
            self.datasets['val'], 
            batch_size=self.batch_size, 
            # collate_fn=collate_fn,
            shuffle=True, 
            drop_last=True,
            pin_memory=pin_memory)
    
        self.epoch = 0

        self.model = AudioLSTM(n_feature=168, out_feature=3)
        self.model.to(self.device)
        print(self.model)
        self.model_path = "model.pt"

        lr = 0.01
        weight_decay = 0.0001

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.criterion = torch.nn.CrossEntropyLoss()


        log_dir = 'logs/' + datetime.now().strftime('%B%d_%H_%M_%S')
        self.writer = SummaryWriter(log_dir)

    def log_scalars(self, global_tag, metric_dict, global_step):

        for tag, value in metric_dict.items():
            self.writer.add_scalar(f"{global_tag}/{tag}", value, global_step)

    def train(self, 
              epoch, 
              loader,
              log_interval=1
            ):

        metric_dict = {}

        self.model.train()
        correct = 0
        y_pred, y_target = [], []
        with tqdm(loader, unit="batch", leave=True) as tepoch:
            for batch_idx, (data, target) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")
                data = data.to(self.device)
                target = target.to(self.device)

                # print(f"Data: {data}, Shape: {data.shape}, target: {target}, shape: {target}")

                # data = torch.tensor(data)
                # target = torch.tensor(target)

                self.optimizer.zero_grad()
                output, hidden_state = self.model(data, self.model.init_hidden(self.batch_size))
                
                loss = self.criterion(output, target)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()

                pred = torch.max(output, dim=1).indices
                correct += pred.eq(target).cpu().sum().item()
                y_pred = y_pred + pred.tolist()
                y_target = y_target + target.tolist()

                tepoch.set_postfix(loss=loss.item(), accuracy=(100. * correct / (self.batch_size*(batch_idx+1))), refresh=True)

                # with open("predictions.txt", "a") as f:
                #     for i in y_target:
                #         f.write(f"Target: {y_target[i]}, Prediction: {y_pred[i]}")

                # if batch_idx % log_interval == 0: #print training stats

                #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         epoch, batch_idx * len(data), len(loader.dataset),
                #         100. * batch_idx / len(loader), loss))

        metric_dict["Loss"] = loss
        metric_dict["Accuracy"] =  100. * correct / len(loader.dataset)         
        
        self.log_scalars("Train", metric_dict, epoch)
                
            
                

    def test(self,
             epoch, 
             loader, 
             log_interval=1):

        metric_dict = {}

        self.model.eval()
        correct = 0
        y_pred, y_target = [], []
        with tqdm(loader, unit="batch", leave=True) as tepoch:
            for batch_idx, (data, target) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {self.epoch}")
                data = data.to(self.device)
                target = target.to(self.device)
                
                output, hidden_state = self.model(data, self.model.init_hidden(self.batch_size))

                loss = self.criterion(output, target)
                
                pred = torch.max(output, dim=1).indices
                correct += pred.eq(target).cpu().sum().item()
                y_pred = y_pred + pred.tolist()
                y_target = y_target + target.tolist()

                # if batch_idx % log_interval == 0: #print training stats

                #     print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
                #         correct, len(loader.dataset),
                #         100. * correct / len(loader.dataset)))
                tepoch.set_postfix(loss=loss.item(), accuracy=(100. * correct / (self.batch_size*(batch_idx+1))), refresh=True)

        
        metric_dict["Loss"] = loss
        metric_dict["Accuracy"] =  100. * correct / len(loader.dataset)

        self.log_scalars("Eval", metric_dict, epoch)
        
        
    def train_model(self):


        log_interval = 1
        for epoch in range(1, 41):
            # scheduler.step()
            self.train(self.model, epoch, self.train_loader, log_interval)
            self.test(self.model, epoch, self.eval_loader, log_interval) 

    def save_model(self):
        torch.save(self.model, self.model_path)
