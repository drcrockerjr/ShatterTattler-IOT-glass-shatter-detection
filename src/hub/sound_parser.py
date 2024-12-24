import os
import wave
from pydub import AudioSegment
import torch, torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import librosa
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


data_root = f"../../data/VOICEe_clean/"
assert os.path.exists(data_root), f"VOICe Dataset path doesnt exist"


class SEDetectionDataset(Dataset):
    def __init__(self, train_file, test_file, processed_dir):
        
        self.file_names =[]
        self.labels = []
        self.sound_times =[]
        
        self.train_annotations = train_file
        self.test_annotations = test_file

        self.event_num = 0


    # def __getitem__(self, index):


def preprocess(num_synth=207, 
                data_root = f"../../data/VOICEe_clean/"
                ):

    assert os.path.exists(data_root), f"VOICe Dataset path doesnt exist"

    # events = []
    # labels = []

    # i = 0
    event_n = 0

    annotations_root = os.path.join(data_root, "annotation/")
    audio_root = os.path.join(data_root, "audio/")

    new_path = os.path.join(data_root, "new_audio")
    os.makedirs(new_path, exist_ok=True)

    ppth = os.path.join(data_root, "preprocessed/")
    os.makedirs(ppth, exist_ok=True)

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

            old_file = os.path.join(audio_root, f"synthetic_{str(syn_n).zfill(3)}.wav")

        
            new_file = os.path.join(new_path, f"synthetic_{str(syn_n).zfill(3)}_s16le.wav")
            os.system(f'ffmpeg -i {old_file} -c:a pcm_s16le {new_file}')

            assert os.path.exists(new_file), f"Audio file does not exist: {new_file}"

            
            # wfile = os.path.join(audio_root, f"synthetic_204_s16le.wav")

            wform, Fs = torchaudio.load(new_file, format="wav")



            for start_time, end_time, label in events:
                # labels.append(label)
                
                print(f"Event {event_n} Start: {int(float(start_time)*Fs)}, End: {int(float(end_time)*Fs)}")

                event_wform = wform[:, int(float(start_time)*Fs): int(float(end_time)*Fs)]

                event_path = os.path.join(ppth, f"{label}_{syn_n}_{event_n}.wav")

                torchaudio.save(event_path, event_wform, Fs)
                event_n += 1
        else:
            print(f"file {fp} does not exist")
        

if __name__=="__main__":
    preprocess()     

def get_wav_info(filename):
    with wave.open(filename, 'rb') as wav_file:
        params = wav_file.getparams()
        return params
    
def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")


def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)


# wave_num = 204

# events = []
# labels = []
# annotations_root = os.path.join(data_root, "annotation/")
# audio_root = os.path.join(data_root, "audio/")

# with open(os.path.join(annotations_root, f"synthetic_{wave_num}.txt"), "r") as file:
#     for line in file:
#         event_info = line.split()
#         events.append(event_info)

# print(events)
# # data/VOICEe_clean/audio/synthetic_204.wav
# # wfile = os.path.join(audio_root, f"synthetic_204.wav")
# wfile = os.path.join(audio_root, f"synthetic_204_s16le.wav")
# print(wfile)

# assert os.path.exists(wfile), f"Audio file does not exist: {wfile}"

# # get_wav_info(wfile)
# # torchaudio.set_audio_backend("soundfile")

# wform, Fs = torchaudio.load(wfile, format="wav")

# ppth = os.path.join(data_root, "preprocessed/")
# os.makedirs(ppth, exist_ok=True)

# # processed_wf = AudioSegment.from_file(wfile, format="wav" )
# # print(processed_wf)
# # processed_wf.export(os.path.join(ppth, "original_processed.wav"), format="wav", parameters=["-acodec", "pcm_s16le"])


# # wform, Fs = torchaudio.load(processed_wf)

# # with wave.open(wf, "rb") as wav:
# #     params = wav.getparams()
# #     frame_rate = wav.getframerate()
# #     n_channels = wav.getnchannels()
# #     sample_width = wav.getsampwidth()
# #     n_frames = wav.getnframes()

# #     audio_data = wav.readframes(n_frames)

# start_dur = 0

# print(f"Fs: {Fs}")

# for i, (start_time, end_time, label) in enumerate(events):
#     labels.append(label)
    
#     print(f"Event {i} Start: {int(float(start_time)*Fs)}, End: {int(float(end_time)*Fs)}")

#     event_wform = wform[:, int(float(start_time)*Fs): int(float(end_time)*Fs)]

#     event_path = os.path.join(ppth, f"{label}_{i}.wav")

#     torchaudio.save(event_path, event_wform, Fs)

#     if i == 22:
#         # plot_spectrogram(event, Fs)
#         # event_specgram = torchaudio.transforms.Spectrogram()(event_wform)
#         # plot_spectrogram(event_specgram, Fs)
#         # print("Shape of spectrogram: {}".format(event_specgram.size()))
        
#         # plt.imshow(event_specgram.log2()[0,:,:].numpy(), cmap='gray')

#         # plt.show(block=True)
#         spectogram = T.Spectrogram(n_fft=512)

#         spec = spectogram(event_wform)

#         fig, axs = plt.subplots(2, 1)
#         plot_waveform(event_wform, Fs, title="Original waveform", ax=axs[0])
#         plot_spectrogram(spec[0], title="spectrogram", ax=axs[1])
#         fig.tight_layout()

#         plt.show()




# # # Load VOICe Dataset
# # dataset = load_dataset("VOICe")  # type: ignore # Extract raw .wav files from the dataset
# # raw_wav_files = extract_wav_files(dataset) # type: ignore

# # # Convert .wav Files to Mel Spectrograms
# # mel_spectrograms = []
# # for wav_file in raw_wav_files:
# #     mel_spec = convert_to_mel_spectrogram(wav_file)  # type: ignore # Compute Mel Spectrogram
# #     mel_spectrograms.append(mel_spec)

# # # Convert Mel Spectrograms to Torch Tensors
# # torch_tensors = []
# # for mel_spec in mel_spectrograms:
# #     tensor = convert_to_torch_tensor(mel_spec)  # type: ignore # Normalize and prepare tensors for LSTM
# #     torch_tensors.append(tensor)

# # # Initialize LSTM Model
# # model = initialize_lstm_model() # type: ignore # Define the LSTM model architecture

# # # Define Loss Function and Optimizer
# # loss_function = define_loss_function() # type: ignore # Specify loss calculation method
# # optimizer = define_optimizer(model) # type: ignore # Set up optimizer for weight updates

# # # Training and Validation Loop
# # for epoch in range(num_epochs): # type: ignore
# #     for inputs, labels in get_training_batches(torch_tensors): # type: ignore # Iterate over data batches
# #         # Forward Pass
# #         outputs = model(inputs)  # Compute model outputs
        
# #         # Calculate Loss
# #         loss = loss_function(outputs, labels)  # Compute the loss
        
# #         # Backpropagation and Weight Update
# #         if training_mode:# type: ignore  # Only backpropagate during training
# #             optimizer.zero_grad()  # Clear gradients
# #             loss.backward()  # Backpropagate gradients
# #             optimizer.step()  # Update model weights

# #             # Log Metrics
# #             log_metrics(loss, outputs, labels) # type: ignore # Record loss and accuracy

# #         if validation_mode:# type: ignore  # Only backpropagate during training
# #             optimizer.zero_grad()  # Clear gradients

# #             # Log Metrics
# #             log_metrics(loss, outputs, labels) # type: ignore # Record loss and accuracy
            






# # def run_hub():
# #     # Connect to the Hub Bluetooth Dongle
# #     audio_data = get_audio_from_bluetooth() # type: ignore # Stream raw audio from the Bluetooth device

# #     # Convert raw audio to Mel Spectrograms
# #     mel_spec = convert_to_mel_spectrogram(audio_data) # type: ignore # Compute Mel Spectrogram images

# #     # Convert Mel Spectrograms to Torch Tensors
# #     input_tensor = convert_to_torch_tensor(mel_spec) # type: ignore # Normalize and prepare tensor for LSTM

# #     # Run the LSTM Model for inference
# #     prediction, confidence = lstm_model_inference(input_tensor) # type: ignore # Get model prediction and confidence

# #     # Check if Glass Shatter Event is Detected
# #     if prediction:  # If glass shatter is detected
# #         # Determine Confidence Value
# #         shatter_event = True  # Set shatter event to True
# #         confidence_value = confidence  # Extract confidence value

# #         # Send System Notification
# #         send_system_notification(shatter_event, confidence_value) # type: ignore # Notify the system with details
# #     else:
# #         # Log Nothing if no shatter is detected
# #         pass
# # import time
# # import smtplib
# # import MIMEMultipart, MIMEText

# # # Initialize Notification System
# # smtp_config = {
# #     "server": "smtp.example.com", 
# #     "port": 587, 
# #     "email": "your_email@example.com",  # Sender email address
# #     "password": "your_email_password",  # Sender email password
# # }  

# # size_limit = 1 * 1024 * 1024  # Set maximum message size to 1MB

# # # Format Notification Content
# # timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) 
# # device_id = "Device1234"  # unique identifier of the edge device
# # auxiliary_data = {"Battery Level": "85%", "Signal Strength": "Strong"}  

# # # Construct the notification message
# # subject = "Glass Break Detected"  # subject of the email
# # message_body = (
# #     f"Alert: A glass break event was detected.\n"
# #     f"Time: {timestamp}\n"
# #     f"Device ID: {device_id}\n"
# #     f"Auxiliary Info: {auxiliary_data}\n"
# # ) 

# # # Ensure message size does not exceed 1MB
# # if len(message_body.encode('utf-8')) > size_limit:
# #     raise ValueError("Notification size exceeds 1MB limit.")  # Validate message size

# # # Send Email Notification
# # try:
# #     # Connect to the SMTP server
# #     with smtplib.SMTP(smtp_config['server'], smtp_config['port']) as server:
# #         server.starttls()  # Enable TLS encryption
# #         server.login(smtp_config['email'], smtp_config['password'])  # Authenticate sender email

# #         # Create the email message
# #         msg = MIMEMultipart()  # Initialize a multipart message
# #         msg["From"] = smtp_config['email']  # Specify the sender
# #         msg["To"] = "recipient_email@example.com"  # Specify the recipient email
# #         msg["Subject"] = subject
# #         msg.attach(MIMEText(message_body, "plain"))  # Attach the message body

# #         # Send the email
# #         server.send_message(msg)  # Send the email message
# #         print(f"Email sent to recipient_email@example.com")  # Confirm email sent
# # except Exception as e:
# #     print(f"Failed to send email: {e}") 