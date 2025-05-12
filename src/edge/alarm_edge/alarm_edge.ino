//server
#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <BLEServer.h>
#include <string.h>
#include <stdio.h>

//BLE Server name
#define bleServerName "XIAOESP32S3_BLE_ALARM"
#define SERVICE_UUID "148ef5d4-f17a-45c0-b38a-ab0344c4fbc6"
#define CHAR_UUID "616e1a1b-b7e4-4277-a230-5af28c1201a6"

#define ALARM_PIN 2

// BLE Setup
BLECharacteristic *pCharacteristic;
bool deviceConnected = false;

//BLE Callback funcs
/*OG
class MyServerCallbacks: public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) {
    deviceConnected = true;
  };
  
  void onDisconnect(BLEServer* pServer) {
    deviceConnected = false;
  }
};*/
class MyCharacteristicCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *pChar)   {
    String value = pChar->getValue();    // Arduino String
    //if (value.length() < 2) return;
    if (!value) return;

    if (value.length() >= 1) {
      // uint8_t low  = (uint8_t)value[0];
      // uint8_t high = (uint8_t)value[1];
      // uint16_t v   = (uint8_t)value[0] | ((uint8_t)value[1] << 8);

      Serial.print("VALUE RECEIVED: ");
      Serial.print((uint8_t)value[0]);
      Serial.print

    

      if ((uint8_t)value[0] == 1) {
        digitalWrite(ALARM_PIN, HIGH);
        digitalWrite(LED_BUILTIN, HIGH);
        Serial.println("Alarm TRIGGERED: GPIO HIGH");
      } else {
        digitalWrite(ALARM_PIN, LOW);
        digitalWrite(LED_BUILTIN, LOW);
        Serial.println("Alarm CLEARED: GPIO LOW");
      }
    }
  }
};



void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  pinMode(ALARM_PIN, OUTPUT);
  digitalWrite(ALARM_PIN, LOW);

  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);


  // Start Bluetooth server and handle setup
  BLEDevice::init(bleServerName);
  BLEServer *pServer = BLEDevice::createServer();


  BLEService *pService = pServer->createService(SERVICE_UUID); //created service with custom made UUID
  pCharacteristic = pService->createCharacteristic(
    BLEUUID(CHAR_UUID),
    BLECharacteristic::PROPERTY_WRITE
  );

  pCharacteristic->setCallbacks(new MyCharacteristicCallbacks());

  
  pService->start();
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(pService->getUUID());
  pAdvertising->setScanResponse(true);
  pAdvertising->setMinPreferred(0x0);
  pAdvertising->setMinPreferred(0x1F);
  BLEDevice::startAdvertising();



  //pCharacteristic->addDescriptor(new BLE2902());
  

}

void loop() {
  // put your main code here, to run repeatedly:
  delay(100);

}
