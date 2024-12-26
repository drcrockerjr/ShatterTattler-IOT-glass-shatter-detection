import os
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from torch.utils.data import Subset
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_LABELS = ["gunshot", "babycry", "glassbreak"]

def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(DATA_LABELS.index(word))


def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return DATA_LABELS[index]

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

class SEDetectionDataset(Dataset):
    def __init__(self, csv_file="audio_info.csv", data_root = f"../../data/VOICe_clean/"):
        
        self.file_names = []
        self.labels = []

        self.max_event_length = 0
        
        self.data_root = data_root
        self.processed_root = os.path.join(self.data_root, "preprocessed/")

        csv_path = os.path.join(self.processed_root, csv_file)

        self.event_num = 0
        csvData = pd.read_csv(csv_path)

        for i in range(1, len(csvData)):
            self.labels.append(csvData.at[i, "label"])

            if csvData.at[i, "length"] > self.max_event_length:
                self.max_event_length = csvData.at[i, "length"]

            self.file_names.append(csvData.at[i, "file_name"])

        # print(self.labels)
        # print(self.file_names)
        print(f"Max Event wf length: {self.max_event_length}")

        self.max_seq_len = 200


    def __getitem__(self, index):

        wf_path = os.path.join(self.processed_root, "event_audio", self.file_names[index])
        wf, sample_rate = torchaudio.load(wf_path, normalize=True)
        # print_stats(wf)

        n_fft = 1024
        win_length = None
        hop_length = 512

        # define transformation
        spectrogram = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
        )

        spec = spectrogram(wf)

        # print_stats(spec)
        # plot_spectrogram(spec[0], title='torchaudio')

        soundData = torch.mean(wf, dim=0, keepdim=True)
        # tempData = torch.zeros([1, self.max_event_length])

        # if soundData.numel() < self.max_event_length:
        #     tempData[:, :soundData.numel()] = soundData
        # else:
        #     tempData = soundData[:, :self.max_event_length]

        # soundData = tempData

        mel_specgram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate
            # n_fft=1024,
            # hop_length=512
            )(soundData)  # (channel, n_mels, time)
        mel_specgram_norm = (mel_specgram - mel_specgram.mean()) / mel_specgram.std()

        mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate
            # n_mfcc=40
            )(soundData)  # (channel, n_mfcc, time)
        mfcc_norm = (mfcc - mfcc.mean()) / mfcc.std()
        # spectogram = torchaudio.transforms.Spectrogram(sample_rate=sample_rate)(soundData)
        feature = torch.cat([mel_specgram, mfcc], axis=1)
        # print(f"Feature: {feature}, Shape: {feature.shape}")

        feature = feature[0].permute(1, 0)

        if feature.size(0) > self.max_seq_len:
            feature = feature[:self.max_seq_len, :]  # Truncate
        else:
            padding = torch.zeros(self.max_seq_len - feature.size(0), feature.size(1))
            feature = torch.cat([feature, padding], dim=0)  # Pad

        return feature, label_to_index(self.labels[index])
    
    def __len__(self):
        return len(self.file_names)