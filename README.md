# ShatterTattler

ShatterTattler is an embedded audio signal analysis platform designed to run deep learning sound event classification in a more decentralized manner. The system is specifically built for low power, glass shatter event detection in residential or commercial environments. Due to the flexible deep learning approach the system can be trained and optimized for real time acoustic sensing applications such as environmental monitoring, industrial condition assessment, and anomaly detection.

# Overview

ShatterTattler integrates high-fidelity audio capture with efficient preprocessing and inference. It leverages the ESP32-S3 microcontroller for connectivity and control, with a PCM1841-Q1 quad channel ADC for microphone audio input from multiple sources. Audio Telemetry data is then transported vio BLE 5.0 from the edge ESP32 to an Nvidia Jetson Orin Nano where the bulk of the heavy processing occurs. Upon data arrival, preprocessing (filtering, spectral analysis, feature extraction) is performed on the Jetson upon data arrival device CPU before LSTM model inference occurs on the device GPU. A general overview is described in this image: 
<img width="1104" height="622" alt="image" src="https://github.com/user-attachments/assets/48dbb552-07d4-49c1-9b34-3cc0df519917" />


# Hardware Components
## ESP32-S3

- Integrated Wi-Fi + Bluetooth LE 5.0
- Up to 512KB SRAM and external PSRAM support
- PCM1841-Q1 (TI Burr-Brown™ Audio ADC) Quad-channel, 32-bit, up to 192kHz sampling
- 123dB dynamic range (with DRE enabled)
- Interfaces: TDM, I²S, or Left-Justified formats
- Integrated MICBIAS (2.75V, 20mA drive) for powering microphones
- Built-in PLL, DC-removal HPF, and configurable digital filters
- Low power consumption (~18–25mW/channel at 48kHz)

## Software Stack

- Arduino Framework (ESP-IDF optional): for ESP32-S3 development
- BLE GATT Server: for wireless configuration and streaming metadata
- DSP Preprocessing: band-pass, low-pass, amplification
- ML Inference: models trained in PyTorch/TensorFlow Jetson runs model directly

# Key Features

- Multi channel audio capture and preprocessing at the edge
- Jeston device neural network inference (CNN, LSTM, or Transformer-lite models)
- BLE-based telemetry and configuration
- Modular design for integration into larger IoT sensing systems
