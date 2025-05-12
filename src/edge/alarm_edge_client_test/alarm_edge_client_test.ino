//client
#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEClient.h>
#include <BLEServer.h>
#include <Arduino.h>

BLEClient*  pClient;
bool doconnect = false;

//BLE Server name (the other ESP32 name running the server sketch)
#define bleServerName "XIAOESP32S3_BLE_ALARM"
#define SERVICE_UUID "148ef5d4-f17a-45c0-b38a-ab0344c4fbc6"
#define CHAR_UUID "616e1a1b-b7e4-4277-a230-5af28c1201a6"

//Address of the peripheral device. Address will be found during scanning...
static BLEAddress *pServerAddress;

/* UUID's of the service, characteristic that we want to read*/
// BLE Service
static BLEUUID serviceUUID(SERVICE_UUID);
static BLEUUID charUUID(CHAR_UUID);

BLERemoteCharacteristic* pCharacteristic = nullptr;


//Callback function that gets called, when another device's advertisement has been received
class MyAdvertisedDeviceCallbacks: public BLEAdvertisedDeviceCallbacks {
  void onResult(BLEAdvertisedDevice advertisedDevice) {
    if (advertisedDevice.getName() == bleServerName) { //Check if the name of the advertiser matches
      advertisedDevice.getScan()->stop(); //Scan can be stopped, we found what we are looking for
      pServerAddress = new BLEAddress(advertisedDevice.getAddress()); //Address of advertiser is the one we need
      Serial.println("Device found. Connecting!");
    }
  }
};


void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
 
  Serial.println("Starting BLE client...");

  BLEDevice::init("XIAOESP32S3_Client");
  BLEScan* pBLEScan = BLEDevice::getScan();
  pBLEScan->setAdvertisedDeviceCallbacks(new MyAdvertisedDeviceCallbacks());
  pBLEScan->setActiveScan(true);
  pBLEScan->start(30);
  pClient = BLEDevice::createClient();

  // Connect to the remove BLE Server.
  pClient->connect(*pServerAddress);
  Serial.println(" - Connected to server");

    // Obtain a reference to the service we are after in the remote BLE server.
  BLERemoteService* pRemoteService = pClient->getService(serviceUUID);

  // Obtain a reference to the service we are after in the remote BLE server.
  pCharacteristic = pRemoteService->getCharacteristic(charUUID);
  if (pCharacteristic == nullptr) {
    Serial.print("Failed to find our characteristic UUID");
    return;
  }

}

void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available() > 0) {
    char input = Serial.read();
    uint8_t data[2];

    if (input == '1') {
      data[0] = 1;
      //data[1] = 0;
      Serial.println("Sent 1 to trigger alarm");
    } else if (input == '0') {
      data[0] = 0;
      //data[1] = 0;
      Serial.println("Sent 0 to clear alarm");
    } else {
      Serial.println("Invalid input: please send '1' or '0'");
      return;
    }

    // Check if connected before writing
    if (pClient && pClient->isConnected() && pCharacteristic) {
      Serial.println("Writing Value!");
      pCharacteristic->writeValue(data, 2);
    } else {
      Serial.println("Not connected to server");
    }
  }

}
