import os
import logging
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch.utils.data import Subset
import time
from datetime import datetime

from dataset import index_to_label

from train import ModelTrainer
from dataset import SEDetectionDataset
from notification import notify_user, AlertCode




logger = logging.getLogger(__name__)



def main():

    trainer = ModelTrainer(data_root=f"../../data/VOICe_clean/", preprocess_data=False)

    dataset = SEDetectionDataset()

    # Get subset will ll values
    small_set = Subset(dataset=dataset, indices=list(range(25)))

    print(f"small_set size: {len(small_set)}")

    batch_size = 1

    eval_loader = torch.utils.data.DataLoader(
            small_set, 
            batch_size=batch_size, 
            # collate_fn=collate_fn,
            shuffle=True, 
            drop_last=True)

    # trainer.train_model()
    device = torch.device('cuda')

    print(f" Loading state of existing saved model at path: {trainer.state_path}")
    trainer.load_state()

    trainer.test(trainer.eval_loader)

    start_t = time.time()
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    
    # From edge device
    uuid = 417
    flag = False



    for idx, (data, target) in enumerate(eval_loader):

        data, target = data.to(device), target.to(device)

        size_in_bytes = data.element_size() * data.numel()

        with open("checkoff.txt", "a") as f:

            f.write(f"[{idx}]  Data Shape: {data.shape}, dtype: {data.dtype}, Datasize: {size_in_bytes}---> Target: {target.item()} = {index_to_label(target.item())}\n")


            output, _ = trainer.model(data, trainer.model.init_hidden(1))

            prediction = torch.max(output, dim=1).indices
            # print(f"[{idx}]Preiction: {prediction}, Shape: {prediction.shape}\n")

            f.write(f"[{idx}] Prediction: {index_to_label(prediction.item())}  vs  Target: {index_to_label(target.item())}\n\n")

            if index_to_label(prediction.item()) == "glassbreak":
                flag = True
            if flag == True:
                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"Glass break happended from device: {uuid}, Flag: {flag}, Timestamp: {timestamp_str}\n\n")

                notify_user(AlertCode.GLASS_BREAK, "4pm", "0440")
                flag = False

            if idx == len(eval_loader):
                f.write(f"\n\n Finished after: {time.time() - start_t}")



    with open("checkoff.txt", "a") as f:
        f.write(f"\n\n Finished after: {time.time() - start_t}")

main()

