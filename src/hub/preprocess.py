import os
import wave
import torch, torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import librosa
import matplotlib.pyplot as plt
import pandas as pd

def preprocess(
    num_synth: int = 40,
    data_root: str = f"../../data/VOICe_clean/"
):
    """
    Read synthetic audio files and their annotation timestamps,
    convert each WAV to 16-bit little-endian, then slice out
    each labeled event segment into its own file and record metadata.
    """
    # Ensure the dataset directory exists
    assert os.path.exists(data_root), f"VOICe Dataset path doesn’t exist: {data_root}"

    # Paths for annotation and raw audio
    annotations_root = os.path.join(data_root, "annotation/")
    audio_root       = os.path.join(data_root, "audio/")

    # Create preprocessed directories if missing
    preproc_root     = os.path.join(data_root, "preprocessed/")
    os.makedirs(preproc_root, exist_ok=True)
    new_audio_pth    = os.path.join(preproc_root, "new_audio")
    os.makedirs(new_audio_pth, exist_ok=True)
    event_audio_pth  = os.path.join(preproc_root, "event_audio")
    os.makedirs(event_audio_pth, exist_ok=True)

    # DataFrame to store metadata for each extracted event
    df = pd.DataFrame(columns=["synthetic_num", "file_name", "label", "length"])

    event_n = 0  # Counter for the total number of events processed

    # Loop over each synthetic file index
    for syn_n in range(num_synth + 1):
        # Construct annotation file path for this synthetic index
        ann_path = os.path.join(
            annotations_root, f"synthetic_{syn_n:03d}.txt"
        )
        if not os.path.isfile(ann_path):
            print(f"Annotation file does not exist: {ann_path}")
            continue

        # Read all lines of the annotation file, splitting into [start, end, label]
        events = []
        try:
            with open(ann_path, "r") as f:
                for line in f:
                    events.append(line.split())
        except IOError:
            print(f"Could not read annotation file: {ann_path}")
            continue

        # Convert the raw WAV to signed 16-bit little-endian PCM using ffmpeg
        raw_wav = os.path.join(audio_root, f"synthetic_{syn_n:03d}.wav")
        out_wav = os.path.join(new_audio_pth, f"synthetic_{syn_n:03d}_s16le.wav")
        os.system(f'ffmpeg -y -i "{raw_wav}" -c:a pcm_s16le "{out_wav}"')
        assert os.path.exists(out_wav), f"Converted file missing: {out_wav}"

        # Load the converted waveform
        waveform, Fs = torchaudio.load(out_wav, format="wav")

        # For each annotated event, slice out the corresponding segment
        for start_time, end_time, label in events:
            start_idx = int(float(start_time) * Fs)
            end_idx   = int(float(end_time)   * Fs)
            length    = end_idx - start_idx

            # Record metadata for this event
            df.at[event_n, "synthetic_num"] = syn_n
            df.at[event_n, "label"]         = label
            df.at[event_n, "length"]        = length
            df.at[event_n, "file_name"]     = f"{label}_{syn_n:03d}_{event_n}.wav"

            print(f"Event {event_n} | {label}: {start_idx}–{end_idx} ({length} samples)")

            # Extract the slice from the full waveform
            event_wform = waveform[:, start_idx:end_idx]
            # Save it under preprocessed/event_audio/
            event_path = os.path.join(
                event_audio_pth, f"{label}_{syn_n:03d}_{event_n}.wav"
            )
            torchaudio.save(event_path, event_wform, Fs)

            event_n += 1

    # After processing all events, write the metadata CSV
    csv_out = os.path.join(preproc_root, "audio_info.csv")
    df.to_csv(csv_out, index=False)
    print(f"Preprocessing complete: {event_n} events saved. Metadata → {csv_out}")

if __name__ == "__main__":
    # Entry point: generate 80 synthetic files by default
    number_of_synth_wav = 80
    data_root = "../../data/VOICe_clean/"
    preprocess(
        num_synth=number_of_synth_wav,
        data_root=data_root
    )
