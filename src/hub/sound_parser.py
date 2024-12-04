import os
import wave
from pydub import AudioSegment
from torch import torchaudio

data_root = f"../../data/VOICEe_clean/"
assert os.path.exists(data_root), f"VOICe Dataset path doesnt exist"

wave_num = 204

events = []
labels = []
annotations_root = os.path.join(data_root, "annotation/")
audio_root = os.path.join(data_root, "audio/")

with open(os.path.join(annotations_root, f"synthetic_{wave_num}.txt"), "r") as file:
    for line in file:
        event_info = line.split()
        events.append(event_info)

print(events)

wfile = os.path.join(audio_root, f"synthetic_{wave_num}.wav")
print(wfile)


ppth = os.path.join(data_root, "preprocessed/", f"synthetic_{wave_num}")
os.makedirs(ppth, exist_ok=True)

processed_wf = AudioSegment.from_file(wf, format="wav")
processed_wf.export(os.path.join(ppth, "original_processed.wav"), format="wav", parameters=["-acodec", "pcm_s16le"])


wform, Fs = torchaudio.load(processed_wf)

# with wave.open(wf, "rb") as wav:
#     params = wav.getparams()
#     frame_rate = wav.getframerate()
#     n_channels = wav.getnchannels()
#     sample_width = wav.getsampwidth()
#     n_frames = wav.getnframes()

#     audio_data = wav.readframes(n_frames)

start_dur = 0

for i, (start_time, end_time, label) in enumerate(events):
    labels.append(label)

    event = wform[:, start_time*Fs: end_time*Fs]




