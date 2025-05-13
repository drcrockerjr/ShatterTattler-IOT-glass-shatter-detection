// //server
 #include <BLEDevice.h>
 #include <BLEUtils.h>
 #include <BLE2902.h>
 #include <BLEServer.h>
 #include <iostream>
 #include <array>
 #include <algorithm>

 /* power management */
#include "esp_pm.h"

// #include "freertos/FreeRTOS.h"
// #include "freertos/task.h"

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


#define OVERSAMPLE   4   // how many raw readings per output sample

#define AUDIO_IN_PIN GPIO_NUM_4

// I2S Specific pin declerations
#define SDNZ_PIN GPIO_NUM_4
#define FSYNC_PIN GPIO_NUM_5
#define BCLK_PIN GPIO_NUM_6
#define I2S_AUDIO_IN_PIN GPIO_NUM_43

const bool USE_I2S = true;

static adc_channel_t channel[1] = {ADC_CHANNEL_2};  // Only read GPIO3
static adc_unit_t unit = ADC_UNIT_1;

// BLE Setup
BLECharacteristic *pCharacteristic;
bool deviceConnected = false;                                   // True if device connected to ESP32, False otherwise

//ADC Read setup
int mtu = 0;                                                    // Stores mtu recieved from BLE device
const uint32_t sample_freq = 12000;
const uint32_t NUM_AUDIO_SECOND = 3;                            // Number of seconds of audio to read
const uint32_t NUM_SAMPLES = NUM_AUDIO_SECOND*sample_freq;      // Number of audio samples to read before BLE send
volatile bool thresh_set = false;                               // True if audio threshold met, False otherwise
esp_err_t ret;                                                  // ESP Return Error 
uint32_t ret_num = 0;                                           // Number of bytes returned from ADC read
uint8_t result[EXAMPLE_READ_LEN] = {0};                         // Buffer to store values from one continuous read
adc_continuous_handle_t handle = NULL;
uint32_t data = 0;                                              // Stores audio data for specific channel
// static uint16_t buffer[NUM_SAMPLES];                                   // Stores all audio data fro NUM_SAMPLES total measurements
volatile uint64_t lifetime_audio_evnts = 0;                     // Stors the total number of audio events that have occured over time
static TaskHandle_t s_task_handle;                              // Task handle used for ADC tasks


// Code from: https://github.com/espressif/esp-idf/blob/v5.4.1/examples/bluetooth/nimble/power_save/main/main.c
    // Configure dynamic frequency scaling:
    // automatic light sleep is enabled if tickless idle support is enabled.
esp_pm_config_t pm_config = {
  .max_freq_mhz = 160,
  .min_freq_mhz = 40,
  .light_sleep_enable = true
};
// ESP_ERROR_CHECK( esp_pm_configure(&pm_config) );

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//Background BUILT_IN LED flash speed control functionality
//////////////////////////////////////////////////////////////////////////////////////////////////////////

enum LED_SETTING { VERY_FAST, FAST, SLOW, VERY_SLOW, SOLID_ON, SOLID_OFF };
Ticker ledTicker;
const uint8_t LED_PIN = LED_BUILTIN;
LED_SETTING  ledSetting;

// Timer Variable
unsigned long start_time;
unsigned long send_start;
volatile unsigned long thresh_start;

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// I2S Extrenal ADC Declerations
//////////////////////////////////////////////////////////////////////////////////////////////////////////

static const i2s_port_t   I2S_PORT      = I2S_NUM_0;
static const int          i2s_sample_rate   = 16000;      // your desired rate
const uint32_t            I2S_NUM_SAMPLES = i2s_sample_rate*NUM_AUDIO_SECOND; 
static const size_t       DMA_BUF_LEN   = 512;        // in samples
static const size_t       DMA_BUF_COUNT = 4;
static int32_t           i2s_buffer[I2S_NUM_SAMPLES];

static const i2s_pin_config_t i2s_pin_config = {
  .bck_io_num   = BCLK_PIN,  // bit clock from your ADC
  .ws_io_num    = FSYNC_PIN,  // word (LR) clock
  .data_out_num = I2S_PIN_NO_CHANGE,
  .data_in_num  = I2S_AUDIO_IN_PIN,  // data line from your ADC
};

i2s_config_t i2s_cfg = {
  .mode                 = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX),
  .sample_rate          = i2s_sample_rate,
  .bits_per_sample      = I2S_BITS_PER_SAMPLE_32BIT,            // <-- 32-bit slot
  // .channel_format       = I2S_CHANNEL_FMT_RIGHT_LEFT,           // or RIGHT_LEFT
  .channel_format       = I2S_CHANNEL_FMT_ONLY_LEFT,           // or RIGHT_LEFT

  // .communication_format = I2S_COMM_FORMAT_I2S_MSB, I2S_COMM_FORMAT_I2S
  .communication_format = I2S_COMM_FORMAT_I2S,
  .intr_alloc_flags     = 0,
  .dma_buf_count        = 4,
  .dma_buf_len          = 512,
  .use_apll             = false
};
// (you can also tune gain, APLL, etc)

// i2s_adc_enable(I2S_NUM_0);
// i2s_adc_channel_t ch = ADC1_CHANNEL_3;           // must be ADC1
// i2s_set_adc_mode(ADC_UNIT_1, ch);

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// LED Indication Functions
//////////////////////////////////////////////////////////////////////////////////////////////////////////

void toggle() {
  digitalWrite(LED_PIN, !digitalRead(LED_PIN));
}

void setLED(LED_SETTING s) {
  ledTicker.detach();              // stop any existing callback
  ledSetting = s;
  switch (s) {
    case VERY_FAST: ledTicker.attach_ms(50, toggle); break;
    case FAST:      ledTicker.attach_ms(200, toggle); break;
    case SLOW:      ledTicker.attach_ms(400, toggle); break;
    case VERY_SLOW: ledTicker.attach_ms(1000, toggle); break;
    case SOLID_ON:  digitalWrite(LED_PIN, LOW);         break;
    case SOLID_OFF: digitalWrite(LED_PIN, HIGH);          break;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// BLE Callbacks, task functions, and buffer sending
//////////////////////////////////////////////////////////////////////////////////////////////////////////

class MyServerCallbacks: public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) {
    deviceConnected = true;
    uint16_t bleID = pServer->getConnId();
    pServer->updatePeerMTU(bleID, 524);
    // uint16_t test = pServer->getPeerMTU(bleID);

    // Serial.printf("PEER MTU = %d", test);


  };
  
  void onDisconnect(BLEServer* pServer) {
    deviceConnected = false;
  }
};

static bool IRAM_ATTR s_conv_done_cb(adc_continuous_handle_t handle, const adc_continuous_evt_data_t *edata, void *user_data)
{
    BaseType_t mustYield = pdFALSE;
    //Notify that ADC continuous driver has done enough number of conversions
    vTaskNotifyGiveFromISR(s_task_handle, &mustYield);

    return (mustYield == pdTRUE);
}

static void continuous_adc_init(adc_channel_t *channel, 
                                uint8_t channel_num, 
                                adc_continuous_handle_t *out_handle)
{
    adc_continuous_handle_t handle = NULL;

    adc_continuous_handle_cfg_t adc_config = {
        .max_store_buf_size = 1024,
        .conv_frame_size = EXAMPLE_READ_LEN,
    };
    ESP_ERROR_CHECK(adc_continuous_new_handle(&adc_config, &handle));

    adc_continuous_config_t dig_cfg = {
        .sample_freq_hz = sample_freq,
        .conv_mode = EXAMPLE_ADC_CONV_MODE,
        .format = EXAMPLE_ADC_OUTPUT_TYPE,
    };

    adc_digi_pattern_config_t adc_pattern[SOC_ADC_PATT_LEN_MAX] = {0};
    dig_cfg.pattern_num = channel_num;
    for (int i = 0; i < channel_num; i++) {
        adc_pattern[i].atten = EXAMPLE_ADC_ATTEN;
        adc_pattern[i].channel = channel[i] & 0x7;
        // adc_pattern[i].unit = EXAMPLE_ADC_UNIT; unit
        adc_pattern[i].unit = unit;
        adc_pattern[i].bit_width = EXAMPLE_ADC_BIT_WIDTH;

        Serial.printf("adc_pattern[%d].atten is :%"PRIx8 "\n", i, adc_pattern[i].atten);
        Serial.printf("adc_pattern[%d].channel is :%\n"PRIx8 "\n", i, adc_pattern[i].channel);
        Serial.printf("adc_pattern[%d].unit is :%\n"PRIx8 "\n", i, adc_pattern[i].unit);
    }
    dig_cfg.adc_pattern = adc_pattern;
    ESP_ERROR_CHECK(adc_continuous_config(handle, &dig_cfg));

    *out_handle = handle;
}

adc_continuous_evt_cbs_t cbs = {
    .on_conv_done = s_conv_done_cb,
};

void send_packets(int32_t *buf,
                  int buf_size,
                  int samples_per_send,
                  int send_delay)
{
    const int maxPayloadSize          = BLEDevice::getMTU() - 3;
    const int maxValuesPerNotification = maxPayloadSize / 2;
    uint8_t payload[maxPayloadSize];
    int sentValues = 0;

    while (sentValues < buf_size) {
        // ← use buf_size, not size
        int valuesToSend = min(maxValuesPerNotification,
                               buf_size - sentValues);

        for (int j = 0; j < valuesToSend; ++j) {
            int16_t v = downcastSample(buf[sentValues + j]);
            payload[j*2    ] =  v        & 0xFF;
            payload[j*2 + 1] = (v >> 8)  & 0xFF;
            Serial.printf("Value %d Packed\n", v);
        }

        pCharacteristic->setValue(payload, valuesToSend * 2);
        pCharacteristic->notify();
        sentValues += valuesToSend;

        Serial.printf("Sent %d/%d values, Payload size: %d bytes\n",
                      sentValues, buf_size, valuesToSend * 2);

        
        if ((sentValues % samples_per_send) == 0) {
            vTaskDelay(pdMS_TO_TICKS(send_delay)); 
        } else {
            vTaskDelay(pdMS_TO_TICKS(50));  
        }
    }
}

constexpr int32_t RAW_MAX = 600000000;   // measured amplitude ceiling

int16_t downcastSample(int32_t raw) {
  // 1) clamp to expected ±RAW_MAX
  if      (raw >  RAW_MAX) raw =  RAW_MAX;
  else if (raw < -RAW_MAX) raw = -RAW_MAX;

  // normalize to between -1.0 and +1.0
  float norm = raw / float(INT32_MAX);

  // scale to 16-bit range and cast
  //    INT16_MAX == 32767, INT16_MIN == -32768
  int16_t out = int16_t(norm * INT16_MAX);

  return out;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Interrupt Handlers
//////////////////////////////////////////////////////////////////////////////////////////////////////////

void IRAM_ATTR thesh_isr() {

  if ((millis() - thresh_start > 5000) || (lifetime_audio_evnts == 0)) { //Thresh debounce

    lifetime_audio_evnts++;
    thresh_set = true;
    thresh_start = millis();
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Audio Acquisition via built in ESP32 ADC
//////////////////////////////////////////////////////////////////////////////////////////////////////////

// void readAudio(uint64_t num_samples) {

//   ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

//   char unit[] = EXAMPLE_ADC_UNIT_STR(EXAMPLE_ADC_UNIT);

//   memset(result, 0xcc, EXAMPLE_READ_LEN);
//   memset(buffer, 0, num_samples);

//   buffer[0] = 0xA2;

//   int buffer_num = 0;

//   while (buffer_num < num_samples) {
//     ret = adc_continuous_read(handle, result, EXAMPLE_READ_LEN, &ret_num, 30);
//     if (ret == ESP_OK) {
//         // Serial.printf("TASK", "ret is %x, ret_num is %"PRIu32" bytes\n", ret, ret_num);
//         for (int i = 0; i < ret_num; i += SOC_ADC_DIGI_RESULT_BYTES) {
//             adc_digi_output_data_t *p = (adc_digi_output_data_t*)&result[i];
//             uint32_t chan_num = EXAMPLE_ADC_GET_CHANNEL(p);
//             data = EXAMPLE_ADC_GET_DATA(p);
//             /*Check the channel number validation, the data is invalid if the channel num exceed the maximum channel */
//             // if (chan_num == 2/*< SOC_ADC_CHANNEL_NUM(EXAMPLE_ADC_UNIT)*/) {
//               // Serial.printf("Unit: %s, Channel: %" PRIu32 ", Value: %" PRIu32 "\n", unit, chan_num, data);

//             if (buffer_num < num_samples){
//               buffer[buffer_num + 1] = static_cast<int16_t>(data);
//               // Serial.printf("Static Cast: %d\n", static_cast<int16_t>(data));
//               Serial.printf("%d\n", data);
//               buffer_num ++;    
//             }             

//         }
//         /**
//           * Because printing is slow, so every time you call `ulTaskNotifyTake`, it will immediately return.
//           * To avoid a task watchdog timeout, add a delay here. When you replace the way you process the data,
//           * usually you don't need this delay (as this task will block for a while).
//           */
//         vTaskDelay(10);
//     } else if (ret == ESP_ERR_TIMEOUT) {
//         //We try to read `EXAMPLE_READ_LEN` until API returns timeout, which means there's no available data
//         Serial.println("Timeout occured, not enought data from read");
//         // break;
//     }
//   }

// }

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Audio Acquisition via built in External
//////////////////////////////////////////////////////////////////////////////////////////////////////////

size_t readI2Sraw(size_t samplesWanted) {
  size_t bytesRead = 0;
  // each sample is now 4 bytes
  i2s_read(I2S_PORT,
           i2s_buffer,
           samplesWanted * sizeof(int32_t),
           &bytesRead,
           portMAX_DELAY);
  // return number of 32-bit samples read
  return bytesRead / sizeof(int32_t);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Setup All ESP32 functionalities
//////////////////////////////////////////////////////////////////////////////////////////////////////////


void setup() {

  Serial.begin(9600);
  Serial.setDebugOutput(true);

  bool use_ble = true;
  if(use_ble) {
    BLEDevice::init(bleServerName);

    BLEServer *pServer = BLEDevice::createServer();
    pServer->setCallbacks(new MyServerCallbacks());
    
    BLEService *pService = pServer->createService(SERVICE_UUID); //created service with custom made UUID
    pCharacteristic = pService->createCharacteristic(
      BLEUUID((uint16_t)0x2A57), // Digital Output
      BLECharacteristic::PROPERTY_NOTIFY
    );
    pCharacteristic->addDescriptor(new BLE2902());

    // Set the server's maximum MTU size
    BLEDevice::setMTU(512);
    
    pService->start();
    BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(pService->getUUID());
    pAdvertising->setScanResponse(true);
    pAdvertising->setMinPreferred(0x0);
    pAdvertising->setMinPreferred(0x1F);
    BLEDevice::startAdvertising();

    mtu = BLEDevice::getMTU();
  }
  
  if(!USE_I2S) {
    s_task_handle = xTaskGetCurrentTaskHandle();

    continuous_adc_init(channel, sizeof(channel) / sizeof(adc_channel_t), &handle);

    ESP_ERROR_CHECK(adc_continuous_register_event_callbacks(handle, &cbs, NULL));
    // ESP_ERROR_CHECK(adc_continuous_start(handle));

  } else { //I2S

    pinMode(SDNZ_PIN, OUTPUT); // Set shutdown Pin to high to start up I2c external ADC chip

    ret = i2s_driver_install(I2S_PORT, &i2s_cfg, 0, nullptr);
    ret = i2s_set_pin(I2S_PORT, &i2s_pin_config);

    // gpio_set_pull_mode(GPIO_NUM_43, GPIO_FLOATING);

    // digitalWrite(SDNZ_PIN, HIGH);
    i2s_zero_dma_buffer(I2S_PORT);

    // i2s_start(I2S_PORT);
  }

  // Set mic threshold isr
  pinMode(THESH_PIN, INPUT_PULLUP);
  pinMode(LED_BUILTIN, OUTPUT);

  // Initialize LED_BUILTIN
	attachInterrupt(THESH_PIN, thesh_isr, RISING);
  // digitalWrite(LED_BUILTIN, HIGH);

  setLED(SOLID_ON);

  start_time = millis();

  Serial.println(" Waiting 5 seconds for start up ");
  while ((millis() - start_time) < 5000) { };

  Serial.println(" Start up complete, going to sleep...");

  setLED(SOLID_OFF);

  // thresh_start = millis();
}

void loop(void)
{

  /**
    * This is to show you the way to use the ADC continuous mode driver event callback.
    * This `ulTaskNotifyTake` will block when the data processing in the task is fast.
    * However in this example, the data processing (print) is slow, so you barely block here.
    *
    * Without using this event callback (to notify this task), you can still just call
    * `adc_continuous_read()` here in a loop, with/without a certain block timeout.
    */

  bool run_loop = true;

  // for(int samples = 0; samples <= 6; samples++) {
  while(1){

    // thresh_set = true;
    if(thresh_set) {

      if(USE_I2S)
        digitalWrite(SDNZ_PIN, HIGH); // Startup External ADC if using
      else 
        ESP_ERROR_CHECK(adc_continuous_start(handle)); // Start up continous read for built in adc

      
      // thresh_start = millis(); // Comment out if using treshhold!!!!!
      // Serial.printf("Thresh start at %lu millis\n", thresh_start);
      Serial.printf("Theshold event happend, total since startup: %lu\n", lifetime_audio_evnts);

      // digitalWrite(LED_BUILTIN, LOW); // Turn LED on
      setLED(SOLID_ON); // On Initial Wake up turn on LED

      start_time = millis();
      Serial.println(" Waiting 0.02 seconds for ADC start up ");
      while ((millis() - start_time) < 200) { vTaskDelay(1); };
      vTaskDelay(2000);
  
      // setLED(FAST);
      setLED(SOLID_OFF);

      unsigned long audio_start = millis();

      // readAudio(NUM_SAMPLES);
      // readAudioOversampled();

      if(!USE_I2S) {
        // readAudioWithAvg(NUM_SAMPLES);
      } else {
        // digitalWrite(SDNZ_PIN, HIGH); //
        size_t bytes_read; 
        bytes_read = readI2Sraw(I2S_NUM_SAMPLES);

        Serial.printf("I2S Read %d bytes \n", bytes_read);

      }

      unsigned long audio_read_time = millis() - audio_start;
      Serial.printf("Audio Collected for %u Samples after %lu ms\n", NUM_SAMPLES, audio_read_time);

      Serial.println("Audio samples collected, attempting to notify data via BLE");

      setLED(SLOW);
 
      if(deviceConnected) { // Wait if not connected
        Serial.println("Device is connected, Notifying Audio Buffer Data");

      } else {
        Serial.printf("\nWaiting For Connection ");
        start_time = millis();
        while((!deviceConnected) && (millis() - start_time < 30000)) { // If connection doesnt happen within 30 seconds stop trying
          Serial.printf(".");
          vTaskDelay(250);
        }

        if(!deviceConnected) {
          Serial.printf(" \nDevice didnt connect in time! \n");
          goto final;
        }
      }

      Serial.println("Sending ADC Data to Connected Device");
      send_start = millis();

      setLED(VERY_FAST);

      // populateArray(buffer, NUM_SAMPLES);

      // for(int i = 0; i < NUM_AUDIO_SECOND; i++) {
      //   sendArray(i2s_buffer, i2s_sample_rate, (i*i2s_sample_rate));
      //   vTaskDelay(200);
      // }

      send_packets(i2s_buffer,
                  I2S_NUM_SAMPLES,
                  i2s_sample_rate, // send sample_rate samples per second
                  250); // 250 ms delay per audio packet

      Serial.printf("ADC Data Successfully Sent after %lu ms!\n", millis() - send_start);
      // vTaskDelay(1000);

      thresh_set = false;


final:
      // Print last values in array for testing

      // for (int i = 0; i < NUM_SAMPLES; i++) {
      //   Serial.printf("%d\n",downcastSample(i2s_buffer[i]));
      // }
      // Serial.println("");

      if(USE_I2S)
        digitalWrite(SDNZ_PIN, LOW); // Shutdown External ADC if using
      else 
        ESP_ERROR_CHECK(adc_continuous_stop(handle)); // Stop continous read for built in adc

    }

    // Serial.printf("Outside of threshold conditional\n", millis() - send_start);

    setLED(SOLID_OFF);

  }
  if(!USE_I2S)
    ESP_ERROR_CHECK(adc_continuous_deinit(handle));
}

