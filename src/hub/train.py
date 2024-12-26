import os
import torch
import torch.nn as nn
from model import AudioLSTM
from dataset import SEDetectionDataset, train_val_dataset
from preprocess import preprocess

data_root = f"../../data/VOICe_clean"
# data_root = "/Users/dcrockerjr/Downloads/VOICe_clean"
assert os.path.exists(data_root), f"VOICe Dataset path doesnt exist"

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

def train(model, epoch, loader, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device)
        target = target.to(device)

        # print(f"Data: {data}, Shape: {data.shape}, target: {target}, shape: {target}")

        # data = torch.tensor(data)
        # target = torch.tensor(target)

        

        model.zero_grad()
        output, hidden_state = model(data, model.init_hidden(hyperparameters["batch_size"]))
        
        loss = criterion(output, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if batch_idx % log_interval == 0: #print training stats
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader.dataset),
                100. * batch_idx / len(loader), loss))
            

def test(model, epoch, loader, log_interval):
    model.eval()
    correct = 0
    y_pred, y_target = [], []
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        
        output, hidden_state = model(data, model.init_hidden(hyperparameters["batch_size"]))
        
        pred = torch.max(output, dim=1).indices
        correct += pred.eq(target).cpu().sum().item()
        y_pred = y_pred + pred.tolist()
        y_target = y_target + target.tolist()
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))
    
    
if __name__=="__main__":

    # preprocess() 

    data_root = f"../../data/VOICe_clean/"

    hyperparameters = {"lr": 0.01, "weight_decay": 0.0001, "batch_size": 128, "in_feature": 168, "out_feature": 3}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # csv_path = '/kaggle/input/urbansound8k/UrbanSound8K.csv'
    # file_path = '/kaggle/input/urbansound8k/'

    csv_file="audio_info.csv"

    dataset = SEDetectionDataset(csv_file, data_root)

    datasets = train_val_dataset(dataset=dataset)

    print("Train set size: " + str(len(datasets['train'])))
    print("Test set size: " + str(len(datasets['val'])))

    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}  # needed for using datasets on gpu

    train_loader = torch.utils.data.DataLoader(
        datasets['train'], 
        batch_size=hyperparameters["batch_size"], 
        # collate_fn=collate_fn,
        shuffle=True, 
        drop_last=True)
    
    eval_loader = torch.utils.data.DataLoader(
        datasets['val'], 
        batch_size=hyperparameters["batch_size"], 
        # collate_fn=collate_fn,
        shuffle=True, 
        drop_last=True)

    model = AudioLSTM(n_feature=hyperparameters["in_feature"], out_feature=hyperparameters["out_feature"])
    model.to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['lr'], weight_decay=hyperparameters['weight_decay'])
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    clip = 5  # gradient clipping

    log_interval = 1
    for epoch in range(1, 41):
        # scheduler.step()
        train(model, epoch, train_loader, log_interval)
        test(model, epoch, eval_loader, log_interval) 