// Audio ADC Read Defines
#define EXAMPLE_ADC_UNIT                    ADC_UNIT_1
#define _EXAMPLE_ADC_UNIT_STR(unit)         #unit
#define EXAMPLE_ADC_UNIT_STR(unit)          _EXAMPLE_ADC_UNIT_STR(unit)
#define EXAMPLE_ADC_CONV_MODE               ADC_CONV_SINGLE_UNIT_1
#define EXAMPLE_ADC_ATTEN                   ADC_ATTEN_DB_0
#define EXAMPLE_ADC_BIT_WIDTH               SOC_ADC_DIGI_MAX_BITWIDTH

// #if CONFIG_IDF_TARGET_ESP32 || CONFIG_IDF_TARGET_ESP32S2
// #define EXAMPLE_ADC_OUTPUT_TYPE             ADC_DIGI_OUTPUT_FORMAT_TYPE1
// #define EXAMPLE_ADC_GET_CHANNEL(p_data)     ((p_data)->type1.channel)
// #define EXAMPLE_ADC_GET_DATA(p_data)        ((p_data)->type1.data)
// #else
// #define EXAMPLE_ADC_OUTPUT_TYPE             ADC_DIGI_OUTPUT_FORMAT_TYPE2
// #define EXAMPLE_ADC_GET_CHANNEL(p_data)     ((p_data)->type2.channel)
// #define EXAMPLE_ADC_GET_DATA(p_data)        ((p_data)->type2.data)
// #endif

// #if CONFIG_IDF_TARGET_ESP32    \
//  || CONFIG_IDF_TARGET_ESP32S2  \
//  || CONFIG_IDF_TARGET_ESP32S3
//   #define EXAMPLE_ADC_OUTPUT_TYPE   ADC_DIGI_OUTPUT_FORMAT_TYPE1
//   #define EXAMPLE_ADC_GET_CHANNEL(p_data)  ((p_data)->type1.channel)
//   #define EXAMPLE_ADC_GET_DATA(p_data)     ((p_data)->type1.data)
// #else
//   #define EXAMPLE_ADC_OUTPUT_TYPE   ADC_DIGI_OUTPUT_FORMAT_TYPE2
//   #define EXAMPLE_ADC_GET_CHANNEL(p_data)  ((p_data)->type2.channel)
//   #define EXAMPLE_ADC_GET_DATA(p_data)     ((p_data)->type2.data)
// #endif

#define EXAMPLE_ADC_OUTPUT_TYPE     ADC_DIGI_OUTPUT_FORMAT_TYPE2
#define EXAMPLE_ADC_GET_CHANNEL(p)  ((p)->type2.channel)
#define EXAMPLE_ADC_GET_DATA(p)     ((p)->type2.data)

#define EXAMPLE_READ_LEN 256

#define EXAMPLE_ADC_ATTEN ADC_ATTEN_DB_6

// Audio data threshold detection
#define THESH_PIN 9 // Should be GPIO 9/A10/D10

// //BLE Server defines
#define bleServerName "XIAOESP32S3_BLE_SERVER"
#define SERVICE_UUID "cd3b9869-1af5-4154-b5a1-eba4ff91a946"