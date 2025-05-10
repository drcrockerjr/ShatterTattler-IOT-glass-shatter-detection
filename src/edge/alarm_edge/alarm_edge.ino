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
class MyServerCallbacks: public BLEServerCallbacks{
  void onWrite(BLECharacteristic *pChar) {
    String value = pChar->getValue();    // Arduino String

    if (value.length() >= 2) {
      uint8_t low  = (uint8_t)value[0];
      uint8_t high = (uint8_t)value[1];
      uint16_t v   = low | (high << 8);
    

      if (v == 1) {
        digitalWrite(ALARM_PIN, HIGH);
        Serial.println("Alarm TRIGGERED: GPIO HIGH");
      } else {
        digitalWrite(ALARM_PIN, LOW);
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

  // Start Bluetooth server and handle setup
  BLEDevice::init(bleServerName);
  BLEServer *pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());


  BLEService *pService = pServer->createService(SERVICE_UUID); //created service with custom made UUID
  pCharacteristic = pService->createCharacteristic(
    CHAR_UUID,
    BLECharacteristic::PROPERTY_WRITE
  );

  pCharacteristic->addDescriptor(new BLE2902());
  pService->start();
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(pService->getUUID());
  pAdvertising->setScanResponse(true);
  pAdvertising->setMinPreferred(0x0);
  pAdvertising->setMinPreferred(0x1F);
  BLEDevice::startAdvertising();

}

void loop() {
  // put your main code here, to run repeatedly:
  delay(100);

}
