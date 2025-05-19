import os
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from tqdm import tqdm
from datetime import datetime
from model import AudioLSTM
from dataset import SEDetectionDataset, train_val_dataset
from preprocess import preprocess
from torch.utils.tensorboard import SummaryWriter

from typing import Any, Dict, Optional

class ModelTrainer():
    def __init__(
        self, 
        data_root: str = f"../../data/VOICe_clean/", 
        preprocess_data: bool = False
    ):
        # Confirm the dataset directory exists
        assert os.path.exists(data_root), f"VOICe Dataset path doesn't exist: {data_root}"
        csv_file = "audio_info.csv"

        # Optionally run data augmentation/preprocessing before loading dataset
        if preprocess_data:
            preprocess(num_synth=80, data_root=data_root)

        # Initialize the detection dataset and split into train/validation
        dataset = SEDetectionDataset(csv_file, data_root)
        self.datasets = train_val_dataset(dataset=dataset)

        print(f"Train set size: {len(self.datasets['train'])}")
        print(f"Test set size: {len(self.datasets['val'])}")

        # Device configuration: default to CPU, switch to GPU if available
        self.device = 'cpu'
        num_workers = 0
        pin_memory = False
        self.batch_size = 128

        if torch.cuda.is_available():
            self.device = 'cuda'
            pin_memory = True

        # Create data loaders for training and evaluation
        self.train_loader = torch.utils.data.DataLoader(
            self.datasets['train'],
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        self.eval_loader = torch.utils.data.DataLoader(
            self.datasets['val'],
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        self.epoch = 0  # Track current epoch

        # Initialize the LSTM model for audio event detection
        self.model = AudioLSTM(n_feature=168, out_feature=3)
        self.model.to(self.device)
        print(self.model)

        # Prepare directory to save model checkpoints
        mdl_dir = "saved_model"
        os.makedirs(mdl_dir, exist_ok=True)
        self.state_path = os.path.join(mdl_dir, "model.pt")

        # Optimizer and loss function setup
        lr = 0.01
        weight_decay = 0.0001
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

        # TensorBoard writer for logging metrics
        log_dir = 'logs/' + datetime.now().strftime('%B%d_%H_%M_%S')
        self.writer = SummaryWriter(log_dir)

    def log_scalars(self, global_tag: str, metric_dict: Dict[str, Any], global_step: int):
        """
        Write a dict of scalar metrics to TensorBoard under a given tag.
        """
        for tag, value in metric_dict.items():
            self.writer.add_scalar(f"{global_tag}/{tag}", value, global_step)

    def train(self, loader: torch.utils.data.DataLoader, log_interval: int = 1):
        """
        Run one training epoch over the provided DataLoader.
        """
        self.model.train()  # Set model to training mode
        correct = 0
        y_pred, y_target = [], []

        # Progress bar for batches
        with tqdm(loader, unit="batch", leave=True) as tepoch:
            for batch_idx, (data, target) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {self.epoch}")
                data = data.to(self.device)
                target = target.to(self.device)

                # Zero gradients before backward pass
                self.optimizer.zero_grad()
                output, hidden_state = self.model(
                    data,
                    self.model.init_hidden(self.batch_size)
                )
                loss = self.criterion(output, target)
                loss.backward()

                # Gradient clipping to avoid exploding gradients
                nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()

                # Compute batch accuracy
                pred = torch.max(output, dim=1).indices
                correct += pred.eq(target).cpu().sum().item()
                y_pred.extend(pred.tolist())
                y_target.extend(target.tolist())

                # Update progress bar with current loss and accuracy
                batch_acc = 100. * correct / ((batch_idx + 1) * self.batch_size)
                tepoch.set_postfix(loss=loss.item(), accuracy=batch_acc, refresh=True)

        # Log final epoch metrics
        metric_dict = {
            "Loss": loss.item(),
            "Accuracy": 100. * correct / len(loader.dataset)
        }
        self.log_scalars("Train", metric_dict, self.epoch)

    def test(self, loader: torch.utils.data.DataLoader, log_interval: int = 1):
        """
        Evaluate the model on the validation set.
        """
        self.model.eval()  # Set model to evaluation mode
        correct = 0
        y_pred, y_target = [], []

        with tqdm(loader, unit="batch", leave=True) as tepoch:
            for batch_idx, (data, target) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {self.epoch}")
                data = data.to(self.device)
                target = target.to(self.device)

                output, hidden_state = self.model(
                    data,
                    self.model.init_hidden(self.batch_size)
                )
                loss = self.criterion(output, target)

                pred = torch.max(output, dim=1).indices
                correct += pred.eq(target).cpu().sum().item()
                y_pred.extend(pred.tolist())
                y_target.extend(target.tolist())

                batch_acc = 100. * correct / ((batch_idx + 1) * self.batch_size)
                tepoch.set_postfix(loss=loss.item(), accuracy=batch_acc, refresh=True)

        # Log validation metrics
        metric_dict = {
            "Loss": loss.item(),
            "Accuracy": 100. * correct / len(loader.dataset)
        }
        self.log_scalars("Eval", metric_dict, self.epoch)

    def train_model(self, num_epoch: int = 41, save_interval: int = 5):
        """
        Train and evaluate the model for a set number of epochs,
        saving the state at regular intervals.
        """
        for self.epoch in range(1, num_epoch):
            self.train(self.train_loader)
            self.test(self.eval_loader)

            # Save checkpoint every save_interval epochs
            if self.epoch % save_interval == 0:
                self.save_state()

    def save_state(self):
        """
        Persist model and optimizer state to disk.
        """
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, self.state_path)

    def load_state(self, path: Optional[str] = None):
        """
        Load model and optimizer state from disk.
        """
        ckpt_path = path or self.state_path
        checkpoint = torch.load(ckpt_path)
        self.epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])