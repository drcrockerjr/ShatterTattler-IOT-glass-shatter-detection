// Server-side BLE + ADC + I2S audio capture example

#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <BLEServer.h>
#include <iostream>
#include <array>
#include <algorithm>
#include "esp_pm.h"                 // Power management APIs
#include "driver/adc.h"
#include <Ticker.h>
#include <limits.h>
#include <string.h>
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_adc/adc_continuous.h"
#include "defs.h"
#include "driver/i2s.h"

// How many raw ADC readings to average per output sample
#define OVERSAMPLE   4

// GPIO pin for analog audio input (built-in ADC)
#define AUDIO_IN_PIN GPIO_NUM_4

// I2S pin definitions for external ADC chip
#define SDNZ_PIN          GPIO_NUM_4
#define FSYNC_PIN         GPIO_NUM_5
#define BCLK_PIN          GPIO_NUM_6
#define I2S_AUDIO_IN_PIN  GPIO_NUM_43

// Toggle between built-in ADC vs external I2S ADC
const bool USE_I2S = true;

// Single ADC channel (GPIO3 / ADC1_CHANNEL_2)
static adc_channel_t channel[1] = { ADC_CHANNEL_2 };
static adc_unit_t unit = ADC_UNIT_1;

// BLE characteristic used for sending audio
BLECharacteristic *pCharacteristic;
bool deviceConnected = false;        // Tracks BLE connection state

// ADC read parameters
int mtu = 0;                         // Negotiated MTU size
const uint32_t sample_freq = 12000; // ADC sampling frequency (Hz)
const uint32_t NUM_AUDIO_SECOND = 3; 
const uint32_t NUM_SAMPLES = NUM_AUDIO_SECOND * sample_freq;
volatile bool thresh_set = false;    // Set when threshold interrupt fires
esp_err_t ret;                       // Generic ESP error code
uint32_t ret_num = 0;                // Bytes read from continuous ADC
uint8_t result[EXAMPLE_READ_LEN] = {0};  // Temporary buffer for ADC read
adc_continuous_handle_t handle = NULL;   // Handle for continuous ADC driver
volatile uint64_t lifetime_audio_evnts = 0; // Total threshold events
static TaskHandle_t s_task_handle;       // Task handle for ADC notifications

// Power management configuration (dynamic frequency scaling)
esp_pm_config_t pm_config = {
  .max_freq_mhz = 160,
  .min_freq_mhz = 40,
  .light_sleep_enable = true
};

// LED blink settings and ticker for built-in LED
enum LED_SETTING { VERY_FAST, FAST, SLOW, VERY_SLOW, SOLID_ON, SOLID_OFF };
Ticker ledTicker;
const uint8_t LED_PIN = LED_BUILTIN;
LED_SETTING ledSetting;

// Timing variables for LED and threshold
unsigned long start_time;
unsigned long send_start;
volatile unsigned long thresh_start;

// I2S buffer and config for external ADC capture
static const i2s_port_t I2S_PORT = I2S_NUM_0;
static const int i2s_sample_rate = 16000;
const uint32_t I2S_NUM_SAMPLES = i2s_sample_rate * NUM_AUDIO_SECOND;
static const size_t DMA_BUF_LEN = 512;
static const size_t DMA_BUF_COUNT = 4;
static int32_t i2s_buffer[I2S_NUM_SAMPLES];

static const i2s_pin_config_t i2s_pin_config = {
  .bck_io_num   = BCLK_PIN,
  .ws_io_num    = FSYNC_PIN,
  .data_out_num = I2S_PIN_NO_CHANGE,
  .data_in_num  = I2S_AUDIO_IN_PIN,
};

i2s_config_t i2s_cfg = {
  .mode                 = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX),
  .sample_rate          = i2s_sample_rate,
  .bits_per_sample      = I2S_BITS_PER_SAMPLE_32BIT,
  .channel_format       = I2S_CHANNEL_FMT_ONLY_LEFT,
  .communication_format = I2S_COMM_FORMAT_I2S,
  .intr_alloc_flags     = 0,
  .dma_buf_count        = DMA_BUF_COUNT,
  .dma_buf_len          = DMA_BUF_LEN,
  .use_apll             = false
};

// Toggle the built-in LED on/off
void toggle() {
  digitalWrite(LED_PIN, !digitalRead(LED_PIN));
}

// Set LED behavior based on enumerated speed/solid modes
void setLED(LED_SETTING s) {
  ledTicker.detach();  // Stop any previous ticker callback
  ledSetting = s;
  switch (s) {
    case VERY_FAST: ledTicker.attach_ms(50, toggle); break;
    case FAST:      ledTicker.attach_ms(200, toggle); break;
    case SLOW:      ledTicker.attach_ms(400, toggle); break;
    case VERY_SLOW: ledTicker.attach_ms(1000, toggle); break;
    case SOLID_ON:  digitalWrite(LED_PIN, LOW);       break;
    case SOLID_OFF: digitalWrite(LED_PIN, HIGH);      break;
  }
}

// BLE server connection callbacks
class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) override {
    deviceConnected = true;
    uint16_t connId = pServer->getConnId();
    pServer->updatePeerMTU(connId, 524);  // Request larger MTU for audio
  }
  void onDisconnect(BLEServer* pServer) override {
    deviceConnected = false;
  }
};

// ADC continuous driver callback (notifies reading task)
static bool IRAM_ATTR s_conv_done_cb(
    adc_continuous_handle_t handle,
    const adc_continuous_evt_data_t *edata,
    void *user_data
) {
  BaseType_t mustYield = pdFALSE;
  vTaskNotifyGiveFromISR(s_task_handle, &mustYield);
  return (mustYield == pdTRUE);
}

// Initialize continuous ADC driver with given channels
static void continuous_adc_init(
    adc_channel_t *channel,
    uint8_t channel_num,
    adc_continuous_handle_t *out_handle
) {
  adc_continuous_handle_cfg_t handle_cfg = {
    .max_store_buf_size = 1024,
    .conv_frame_size    = EXAMPLE_READ_LEN,
  };
  ESP_ERROR_CHECK(adc_continuous_new_handle(&handle_cfg, out_handle));

  adc_continuous_config_t cfg = {
    .sample_freq_hz = sample_freq,
    .conv_mode      = EXAMPLE_ADC_CONV_MODE,
    .format         = EXAMPLE_ADC_OUTPUT_TYPE,
  };
  static adc_digi_pattern_config_t pattern[SOC_ADC_PATT_LEN_MAX] = {0};
  cfg.pattern_num = channel_num;
  for (int i = 0; i < channel_num; ++i) {
    pattern[i].atten     = EXAMPLE_ADC_ATTEN;
    pattern[i].channel   = channel[i] & 0x7;
    pattern[i].unit      = unit;
    pattern[i].bit_width = EXAMPLE_ADC_BIT_WIDTH;
    // Debug print pattern config
    Serial.printf("ADC[%d] ch:%d atten:%02x unit:%02x\n",
                  i, pattern[i].channel, pattern[i].atten, pattern[i].unit);
  }
  cfg.adc_pattern = pattern;
  ESP_ERROR_CHECK(adc_continuous_config(*out_handle, &cfg));
}

// Structure to hold ADC event callbacks
adc_continuous_evt_cbs_t cbs = {
  .on_conv_done = s_conv_done_cb,
};

// Send audio samples over BLE in appropriately-sized packets
void send_packets(
    int32_t *buf,
    int buf_size,
    int samples_per_send,
    int send_delay
) {
  const int maxPayloadSize = BLEDevice::getMTU() - 3;
  const int maxValuesPerNotification = maxPayloadSize / 2;
  uint8_t payload[maxPayloadSize];
  int sentValues = 0;

  while (sentValues < buf_size) {
    int valuesToSend = std::min(maxValuesPerNotification, buf_size - sentValues);

    // Pack 16-bit samples into byte payload
    for (int j = 0; j < valuesToSend; ++j) {
      int16_t v = downcastSample(buf[sentValues + j]);
      payload[j*2]     = v & 0xFF;
      payload[j*2 + 1] = (v >> 8) & 0xFF;
      Serial.printf("Packed sample: %d\n", v);
    }

    pCharacteristic->setValue(payload, valuesToSend * 2);
    pCharacteristic->notify();
    sentValues += valuesToSend;

    // Log progress and delay appropriately between packets
    Serial.printf("Sent %d/%d samples\n", sentValues, buf_size);
    int delay_ms = (sentValues % samples_per_send == 0) ? send_delay : 50;
    vTaskDelay(pdMS_TO_TICKS(delay_ms));
  }
}

// Convert 24-bit raw ADC reading down to 16-bit with clipping
int16_t downcastSample(int32_t raw) {
  int32_t pcm24 = raw >> 8;
  constexpr int32_t MAX24 = (1 << 23) - 1;
  pcm24 = std::clamp(pcm24, -MAX24, MAX24);
  float norm = pcm24 / float(MAX24);
  int16_t out = static_cast<int16_t>(norm * INT16_MAX + (norm >= 0 ? 0.5f : -0.5f));
  return out;
}

// Threshold interrupt service routine (debounced)
void IRAM_ATTR thesh_isr() {
  if ((millis() - thresh_start > 5000) || (lifetime_audio_evnts == 0)) {
    lifetime_audio_evnts++;
    thresh_set = true;
    thresh_start = millis();
  }
}

// Read raw 32-bit samples over I2S, return count of samples read
size_t readI2Sraw(size_t samplesWanted) {
  size_t bytesRead = 0;
  i2s_read(I2S_PORT, i2s_buffer,
           samplesWanted * sizeof(int32_t),
           &bytesRead, portMAX_DELAY);
  return bytesRead / sizeof(int32_t);
}

// Arduino setup: initialize serial, BLE, ADC/I2S, interrupts, LED
void setup() {
  Serial.begin(9600);
  Serial.setDebugOutput(true);

  // Initialize BLE server and characteristic
  BLEDevice::init(bleServerName);
  BLEServer *pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());
  BLEService *pService = pServer->createService(SERVICE_UUID);
  pCharacteristic = pService->createCharacteristic(
    BLEUUID((uint16_t)0x2A57),
    BLECharacteristic::PROPERTY_NOTIFY
  );
  pCharacteristic->addDescriptor(new BLE2902());
  BLEDevice::setMTU(512);
  pService->start();
  BLEDevice::startAdvertising();
  mtu = BLEDevice::getMTU();

  // Configure ADC or I2S depending on flag
  if (!USE_I2S) {
    s_task_handle = xTaskGetCurrentTaskHandle();
    continuous_adc_init(channel, 1, &handle);
    ESP_ERROR_CHECK(adc_continuous_register_event_callbacks(handle, &cbs, NULL));
  } else {
    pinMode(SDNZ_PIN, OUTPUT);
    i2s_driver_install(I2S_PORT, &i2s_cfg, 0, nullptr);
    i2s_set_pin(I2S_PORT, &i2s_pin_config);
    i2s_zero_dma_buffer(I2S_PORT);
  }

  // Set up threshold interrupt and turn LED solid on during init
  pinMode(THESH_PIN, INPUT_PULLUP);
  pinMode(LED_BUILTIN, OUTPUT);
  attachInterrupt(THESH_PIN, thesh_isr, RISING);
  if (USE_I2S) digitalWrite(SDNZ_PIN, HIGH);

  setLED(SOLID_ON);
  start_time = millis();
  while ((millis() - start_time) < 5000) { /* wait 5s */ }
  setLED(SOLID_OFF);
}

// Main loop: wait for threshold, capture audio, then send via BLE
void loop() {
  if (thresh_set) {
    // Start ADC or I2S capture
    if (USE_I2S) {
      digitalWrite(SDNZ_PIN, HIGH);
    } else {
      ESP_ERROR_CHECK(adc_continuous_start(handle));
    }

    setLED(SOLID_ON);
    vTaskDelay(pdMS_TO_TICKS(20));
    setLED(SOLID_OFF);

    unsigned long audio_start = millis();
    size_t count = USE_I2S
      ? readI2Sraw(I2S_NUM_SAMPLES)
      : 0;  // built-in ADC path omitted for brevity

    Serial.printf("Captured %u samples in %lu ms\n", NUM_SAMPLES, millis() - audio_start);
    setLED(SLOW);

    // Wait for BLE connection if not already connected
    if (!deviceConnected) {
      unsigned long wait_start = millis();
      while (!deviceConnected && millis() - wait_start < 30000) {
        vTaskDelay(pdMS_TO_TICKS(250));
      }
    }
    send_start = millis();
    setLED(VERY_FAST);
    send_packets(i2s_buffer, I2S_NUM_SAMPLES, i2s_sample_rate, 250);
    Serial.printf("Sent audio in %lu ms\n", millis() - send_start);

    // Clean up and reset threshold
    if (!USE_I2S) ESP_ERROR_CHECK(adc_continuous_stop(handle));
    thresh_set = false;
    setLED(SOLID_OFF);
  }
}
