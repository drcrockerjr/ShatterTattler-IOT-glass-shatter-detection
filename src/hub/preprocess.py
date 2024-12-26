import os
import wave
import torch, torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import librosa
import matplotlib.pyplot as plt
import pandas as pd



def preprocess(num_synth=40, 
                data_root = f"../../data/VOICe_clean/"
                ):

    assert os.path.exists(data_root), f"VOICe Dataset path doesnt exist"

    # events = []
    # labels = []

    # i = 0
    event_n = 0

    annotations_root = os.path.join(data_root, "annotation/")
    audio_root = os.path.join(data_root, "audio/")


    ppth = os.path.join(data_root, "preprocessed/")
    os.makedirs(ppth, exist_ok=True)

    new_audio_pth = os.path.join(ppth, "new_audio")
    os.makedirs(new_audio_pth, exist_ok=True)

    new_evnt_audio_pth = os.path.join(ppth, "event_audio")
    os.makedirs(new_evnt_audio_pth, exist_ok=True)

    df = pd.DataFrame(columns=["synthetic_num", "file_name", "label", "length"])

    for syn_n in range(num_synth+1):

        fp = os.path.join(annotations_root, f"synthetic_{str(syn_n).zfill(3)}.txt")
        if os.path.isfile(fp):
            events = []

            try:

                with open(fp, "r") as file:
                    for line in file:
                        event_info = line.split()
                        events.append(event_info)

            except IOError:
                print (f"Could not read file {fp}")

            old_file = os.path.join(audio_root , f"synthetic_{str(syn_n).zfill(3)}.wav")

        
            new_file = os.path.join(new_audio_pth, f"synthetic_{str(syn_n).zfill(3)}_s16le.wav")
            os.system(f'ffmpeg -i {old_file} -c:a pcm_s16le {new_file}')

            assert os.path.exists(new_file), f"Audio file does not exist: {new_file}"

            
            # wfile = os.path.join(audio_root, f"synthetic_204_s16le.wav")

            wform, Fs = torchaudio.load(new_file, format="wav")



            for start_time, end_time, label in events:
                # labels.append(label)
                df.at[event_n, "synthetic_num"] = syn_n

                df.at[event_n, "label"] = label
                
                df.at[event_n, "length"] = int(float(end_time)*Fs) - int(float(start_time)*Fs)

                print(f"Event {event_n} Start: {int(float(start_time)*Fs)}, End: {int(float(end_time)*Fs)}, Length: {int(float(end_time)*Fs) - int(float(start_time)*Fs)}")

                event_wform = wform[:, int(float(start_time)*Fs): int(float(end_time)*Fs)]

                event_path = os.path.join(new_evnt_audio_pth, f"{label}_{str(syn_n).zfill(3)}_{event_n}.wav")

                df.at[event_n, "file_name"] = f"{label}_{str(syn_n).zfill(3)}_{event_n}.wav"

                torchaudio.save(event_path, event_wform, Fs)
                event_n += 1
        else:
            print(f"file {fp} does not exist")

    df.to_csv(os.path.join(ppth ,'audio_info.csv'))
        