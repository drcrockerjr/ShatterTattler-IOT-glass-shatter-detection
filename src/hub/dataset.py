import os
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, Subset
import pandas as pd
from sklearn.model_selection import train_test_split
from plot_utils import plot_spectrogram
import matplotlib.pyplot as plt

# Labels for different sound event classes
DATA_LABELS = ["gunshot", "babycry", "glassbreak"]

def label_to_index(word: str) -> torch.Tensor:
    """
    Map a label string to its numeric index in DATA_LABELS.
    Returns a 0-dim tensor containing the index.
    """
    return torch.tensor(DATA_LABELS.index(word))

def index_to_label(index: int) -> str:
    """
    Convert a numeric class index back to its label string.
    """
    return DATA_LABELS[index]

def pad_sequence(batch: list[torch.Tensor]) -> torch.Tensor:
    """
    Given a list of [channel, time] tensors, 
    transpose each to [time, channel], pad to the same length,
    then transpose back to [batch, channel, time].
    """
    # Transpose each sample so time is first dimension
    batch = [item.t() for item in batch]
    # Pad sequences to the length of the longest in the batch
    batch = torch.nn.utils.rnn.pad_sequence(
        batch, batch_first=True, padding_value=0.
    )
    # Transpose back to [batch, channel, time]
    return batch.permute(0, 2, 1)

def train_val_dataset(dataset: Dataset, val_split: float = 0.25) -> dict[str, Subset]:
    """
    Split a Dataset into train and validation subsets.
    Returns a dict with 'train' and 'val' keys.
    """
    # Generate indices for splitting
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=val_split)
    return {
        'train': Subset(dataset, train_idx),
        'val':   Subset(dataset, val_idx)
    }

class SEDetectionDataset(Dataset):
    """
    Dataset for single-event audio clips, reading metadata from a CSV
    and generating spectrogram + MFCC features on the fly.
    """
    def __init__(
        self,
        csv_file: str = "audio_info.csv",
        data_root: str = f"../../data/VOICe_clean/"
    ):
        # Paths to preprocessed audio and CSV
        self.data_root = data_root
        self.processed_root = os.path.join(self.data_root, "preprocessed/")
        csv_path = os.path.join(self.processed_root, csv_file)

        # Read CSV metadata
        csvData = pd.read_csv(csv_path)
        self.file_names: list[str] = []
        self.labels: list[str] = []
        self.max_event_length = 0

        # Collect file names, labels, and track maximum clip length
        for i in range(1, len(csvData)):
            length = csvData.at[i, "length"]
            if length > self.max_event_length:
                self.max_event_length = length
            self.file_names.append(csvData.at[i, "file_name"])
            self.labels.append(csvData.at[i, "label"])

        # Maximum sequence length (time frames) for feature padding/truncation
        self.max_seq_len = 200

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Load waveform from preprocessed file
        wf_path = os.path.join(
            self.processed_root, "event_audio", self.file_names[index]
        )
        wf, sample_rate = torchaudio.load(wf_path, normalize=True)

        # Resample to consistent rate
        new_sample_rate = 16000
        wf = T.Resample(sample_rate, new_sample_rate, dtype=torch.float32)(wf)

        # Collapse multi-channel to mono by averaging
        soundData = torch.mean(wf, dim=0, keepdim=True)

        # Compute mel spectrogram and normalize
        mel_specgram = torchaudio.transforms.MelSpectrogram(
            sample_rate=new_sample_rate
        )(soundData)
        mel_specgram = (mel_specgram - mel_specgram.mean()) / mel_specgram.std()

        # Compute MFCC and normalize
        mfcc = torchaudio.transforms.MFCC(
            sample_rate=new_sample_rate
        )(soundData)
        mfcc = (mfcc - mfcc.mean()) / mfcc.std()

        # Concatenate spectrogram and MFCC along channel axis
        feature = torch.cat([mel_specgram, mfcc], dim=1)
        # Rearrange to [time, features]
        feature = feature[0].permute(1, 0)

        # Truncate or pad to fixed number of time frames
        if feature.size(0) > self.max_seq_len:
            feature = feature[:self.max_seq_len, :]
        else:
            pad_len = self.max_seq_len - feature.size(0)
            padding = torch.zeros(pad_len, feature.size(1))
            feature = torch.cat([feature, padding], dim=0)

        # Return feature tensor and label index tensor
        return feature, label_to_index(self.labels[index])

    def __len__(self) -> int:
        return len(self.file_names)

def wav_to_feature(
    wf: torch.Tensor,
    sample_rate: int,
    new_sample_rate: int = 16000
) -> torch.Tensor:
    """
    Convert a raw waveform tensor [channel, time] into a
    padded [max_seq_len, feature_dim] feature tensor,
    concatenating mel spectrogram and MFCC.
    """
    # Resample and collapse to mono
    wf = T.Resample(sample_rate, new_sample_rate, dtype=torch.float32)(wf)
    soundData = torch.mean(wf, dim=0, keepdim=True)

    # Compute normalized mel spectrogram
    mel_specgram = torchaudio.transforms.MelSpectrogram(
        sample_rate=new_sample_rate
    )(soundData)
    mel_specgram = (mel_specgram - mel_specgram.mean()) / mel_specgram.std()

    # Compute normalized MFCC
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=new_sample_rate
    )(soundData)
    mfcc = (mfcc - mfcc.mean()) / mfcc.std()

    # Concatenate and reshape to [time, features]
    feature = torch.cat([mel_specgram, mfcc], dim=1)[0].permute(1, 0)

    # Fixed-length padding/truncation
    max_seq_len = 200
    if feature.size(0) > max_seq_len:
        feature = feature[:max_seq_len, :]
    else:
        pad_len = max_seq_len - feature.size(0)
        feature = torch.cat([
            feature,
            torch.zeros(pad_len, feature.size(1))
        ], dim=0)

    return feature
