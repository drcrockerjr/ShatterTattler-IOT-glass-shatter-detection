import os
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, Subset
import pandas as pd
from sklearn.model_selection import train_test_split
from plot_utils import plot_spectrogram
import matplotlib.pyplot as plt
import torch.nn.functional as F      # <— for conv1d


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

            # wf_path = os.path.join(self.processed_root, "event_audio", csvData.at[i, "file_name"])
            # wf, sample_rate = torchaudio.load(wf_path, normalize=True)

            # new_sample_rate = 16000
            # transform = torchaudio.transforms.Resample(sample_rate, new_sample_rate, dtype=torch.float32)
            # ds_wf = transform(wf)

            # soundData = torch.mean(wf, dim=0, keepdim=True)
            # ds_soundData = torch.mean(ds_wf, dim=0, keepdim=True)

            # mel_specgram = torchaudio.transforms.MelSpectrogram(
            # sample_rate=sample_rate,
            # n_mels=40
            # # hop_length=512
            # )(soundData)  # (channel, n_mels, time)
            
            # mel_specgram2 = torchaudio.transforms.MelSpectrogram(
            # sample_rate=new_sample_rate,
            # n_mels=40
            # # hop_length=512
            # )(ds_soundData)  # (channel, n_mels, time)
            
            # mfcc3 = torchaudio.transforms.MFCC(
            # sample_rate=sample_rate
            # # n_mfcc=40
            # )(soundData)  # (channel, n_mfcc, time)
            
            # mfcc4 = torchaudio.transforms.MFCC(
            # sample_rate=new_sample_rate
            # # n_mfcc=60
            # )(ds_soundData)  # (channel, n_mfcc, time)
            
            # fig, axs = plt.subplots(4,1)
            # print(f"Sample Rate: {sample_rate}")
            # print(f"Mel1 Shape: {mel_specgram.shape} \n Mel2 Shape: {mel_specgram2.shape}\n Mel3 Shape: {mfcc3.shape}\n Mel4 Shape: {mfcc4.shape}")

            # # print_stats(spec)
            # plot_spectrogram(mel_specgram[0], title='mel_spec1', ax=axs[0])
            # plot_spectrogram(mel_specgram2[0], title='mel_spec2', ax=axs[1])
            # plot_spectrogram(mfcc3[0], title='mel_spec3', ax=axs[2])
            # plot_spectrogram(mfcc4[0], title='mel_spec4', ax=axs[3])

            # plt.show(block=True)

        # # print(self.labels)
        # # print(self.file_names)
        # print(f"Max Event wf length: {self.max_event_length}")

        self.max_seq_len = 200


    def __getitem__(self, index):

        wf_path = os.path.join(self.processed_root, "event_audio", self.file_names[index])
        wf, sample_rate = torchaudio.load(wf_path, normalize=True)
        # print_stats(wf)

        new_sample_rate = 16000
        transform = torchaudio.transforms.Resample(sample_rate, new_sample_rate, dtype=torch.float32)
        wf = transform(wf)

        n_fft = 1024
        win_length = None
        hop_length = 512

        # # define transformation
        # spectrogram = T.Spectrogram(
        #     n_fft=n_fft,
        #     win_length=win_length,
        #     hop_length=hop_length,
        #     center=True,
        #     pad_mode="reflect",
        #     power=2.0,
        # )

        # spec = spectrogram(wf)

        

        soundData = torch.mean(wf, dim=0, keepdim=True)
        # tempData = torch.zeros([1, self.max_event_length])

        # if soundData.numel() < self.max_event_length:
        #     tempData[:, :soundData.numel()] = soundData
        # else:
        #     tempData = soundData[:, :self.max_event_length]

        # soundData = tempData

        mel_specgram = torchaudio.transforms.MelSpectrogram(
            sample_rate=new_sample_rate
            # n_mels=40
            )(soundData)  # (channel, n_mels, time)
        
        # mel_specgram2 = torchaudio.transforms.MelSpectrogram(
        #     sample_rate=new_sample_rate,
        #     n_fft=1024
        #     # hop_length=512
        #     )(soundData)  # (channel, n_mels, time)
        
        # fig, axs = plt.subplots(2, 1)

        # print_stats(spec)
        # plot_spectrogram(mel_specgram1, title='mel_spec1', ax=axs[0])
        # plot_spectrogram(mel_specgram2, title='mel_spec2', ax=axs[1])

        # plt.show(block=True)


        mel_specgram_norm = (mel_specgram - mel_specgram.mean()) / mel_specgram.std()

        mfcc = torchaudio.transforms.MFCC(
            sample_rate=new_sample_rate
            # n_mfcc=40
            )(soundData)  # (channel, n_mfcc, time)
        mfcc_norm = (mfcc - mfcc.mean()) / mfcc.std()

        # fig, axs = plt.subplots(4,1)

        # print(f"Mel1 Shape: {mel_specgram.shape} \n Mel2 Shape: {mel_specgram_norm.shape}\n Mel3 Shape: {mfcc.shape}\n Mel4 Shape: {mfcc_norm.shape}")


        # # print_stats(spec)
        # plot_spectrogram(mel_specgram[0], title='mel_spec1', ax=axs[0])
        # plot_spectrogram(mel_specgram_norm[0], title='mel_spec2', ax=axs[1])
        # plot_spectrogram(mfcc[0], title='mel_spec3', ax=axs[2])
        # plot_spectrogram(mfcc_norm[0], title='mel_spec4', ax=axs[3])

        # plt.show(block=True)

        # spectogram = torchaudio.transforms.Spectrogram(sample_rate=sample_rate)(soundData)
        feature = torch.cat([mel_specgram, mfcc], axis=1)
        # print(f"Feature: {feature}, Shape: {feature.shape}")

        feature = feature[0].permute(1, 0)
        # print(f"Feature Shape: {feature.shape}")


        if feature.size(0) > self.max_seq_len:
            feature = feature[:self.max_seq_len, :]  # Truncate
        else:
            padding = torch.zeros(self.max_seq_len - feature.size(0), feature.size(1))
            feature = torch.cat([feature, padding], dim=0)  # Pad

        return feature, label_to_index(self.labels[index])
    
    def __len__(self):
        return len(self.file_names)



def wav_to_feature(wf, sample_rate, new_sample_rate: bool = None, to_mono:bool = True):

    if new_sample_rate != None: 
        # new_sample_rate = 16000
        transform = torchaudio.transforms.Resample(sample_rate, new_sample_rate, dtype=torch.float32)
        wf = transform(wf)  
    else: 
        new_sample_rate = sample_rate
    

    n_fft = 1024
    win_length = None
    hop_length = 512

    max_seq_len = 200

    if wf.ndim == 1:
        soundData = wf.unsqueeze(0)
    else:
        # e.g. stereo: collapse to mono
        soundData = wf.mean(dim=0, keepdim=True)

    # print("resized sounddata")
    # tempData = torch.zeros([1, self.max_event_length])

    # if soundData.numel() < self.max_event_length:
        #     tempData[:, :soundData.numel()] = soundData
    # else:
    #     tempData = soundData[:, :self.max_event_length]

    # soundData = tempData

    # ─── 3) Spike removal via moving average ───
    # window length in samples (e.g. 10 ms)
    window_ms   = 10
    window_size = max(3, int(window_ms * sample_rate / 1000))  # at least 3 samples

    # build a 1-D box filter kernel per channel
    channels = soundData.size(0)
    kernel = torch.ones(channels, 1, window_size, device=soundData.device) / window_size

    # smooth (pad='same')
    x = soundData.unsqueeze(0)  # -> [1, C, T]
    smoothed = F.conv1d(x, kernel, padding=window_size // 2, groups=channels)
    smoothed = smoothed.squeeze(0)  # -> [C, T]

    # detect “spikes” and replace them
    threshold = 0.1  # adjust to your signal’s amplitude scale
    diff = (soundData - smoothed).abs()
    soundData = torch.where(diff > threshold, smoothed, soundData)
    # ──────────────────────────────────────────

    mel_specgram = torchaudio.transforms.MelSpectrogram(
        sample_rate=new_sample_rate
        # n_mels=40
        )(soundData)  # (channel, n_mels, time)

    mel_specgram_norm = (mel_specgram - mel_specgram.mean()) / mel_specgram.std()
    # print("got mel")
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=new_sample_rate
        # n_mfcc=40
        )(soundData)  # (channel, n_mfcc, time)
    mfcc_norm = (mfcc - mfcc.mean()) / mfcc.std()
    # print("got mfcc")
    # fig, axs = plt.subplots(4,1)

    # print(f"Mel1 Shape: {mel_specgram.shape} \n Mel2 Shape: {mel_specgram_norm.shape}\n Mel3 Shape: {mfcc.shape}\n Mel4 Shape: {mfcc_norm.shape}")


    # # print_stats(spec)
    # plot_spectrogram(mel_specgram[0], title='mel_spec1', ax=axs[0])
    # plot_spectrogram(mel_specgram_norm[0], title='mel_spec2', ax=axs[1])
    # plot_spectrogram(mfcc[0], title='mel_spec3', ax=axs[2])
    # plot_spectrogram(mfcc_norm[0], title='mel_spec4', ax=axs[3])

    # plt.show(block=True)

    # spectogram = torchaudio.transforms.Spectrogram(sample_rate=sample_rate)(soundData)
    feature = torch.cat([mel_specgram, mfcc], axis=1)
    # print(f"Feature: {feature}, Shape: {feature.shape}")

    feature = feature[0].permute(1, 0)
    # print(f"Feature Shape: {feature.shape}")
    # print("reshaped tensor")    

    if feature.size(0) > max_seq_len:
        # print("greater than max seq len")
        feature = feature[:max_seq_len, :]  # Truncate
       
    else:
        # print("less than max seq len")
        padding = torch.zeros(max_seq_len - feature.size(0), feature.size(1))
        feature = torch.cat([feature, padding], dim=0)  # Pad\

    # print("finished padding")
    return feature
