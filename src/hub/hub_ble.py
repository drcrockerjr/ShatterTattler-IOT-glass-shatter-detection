
from bleak import BleakScanner, BleakClient
import asyncio
import time
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


logger = logging.getLogger(__name__)


async def run_queue_consumer(queue: asyncio.Queue):
    logger.info("Starting queue consumer")

    while True:
        # Use await asyncio.wait_for(queue.get(), timeout=1.0) if you want a timeout for getting data.
        epoch, data = await queue.get()
        if data is None:
            logger.info(
                "Got message from client about disconnection. Exiting consumer loop..."
            )
            break
        else:
            logger.info("Received callback data via async queue at %s: %r", epoch, data)

async def discover_esp32():

    device = None
    discover_dict = await BleakScanner.discover(adapter="hci1", return_adv=True)

    # dev_dict[0]: BLEDevice, dev_dict[1]: AdvertisementData
    for dev_addr, dev_dict in discover_dict.items():
        logger.info(f"U: {dev_addr}, N: {dev_dict[0].name}, rssi: {dev_dict[0].rssi}")
        
        if dev_dict[0].name == "XIAOESP32S3_BLE_SERVER":
            logger.info(f"found ESP32 at addr: {dev_addr}, UUIDs: {dev_dict[1]}")

            device = dev_dict[0] # For Bleak better to connect with BLEDevice class

    return device


async def run_ble_client(device, queue:asyncio.Queue):

    async def callback_handler(_, data):
        await queue.put((time.time(), data))

    async with BleakClient(device) as client:
        
        if not await client.is_connected():
            logger.error(f"Failed to Connect")
            return

        logger.info(f" Connected to Device: {device.address}")

        logger.info(f" MTU size: {client.mtu_size}")

        services = await client.get_services()
        
        read_char = None
        for service in services:
            logger.info(f"Service: {service.uuid}")
            for characteristic in service.characteristics:
                logger.info(f"  Characteristic: {characteristic.uuid}")
                logger.info(f"    Properties: {characteristic.properties}")

                if 'notify' in characteristic.properties:
                    value = await client.read_gatt_char(characteristic.uuid)
                    logger.info(f"   Value: {value}")

                    read_char = characteristic.uuid

        await client.start_notify(read_char, callback_handler)
        await asyncio.sleep(30.0)

        # await client.stop_notify(read_char)

        await queue.put((time.time(), None))

        logger.info(f"Queueu: {queue._queue}")

async def main():

    queue = asyncio.Queue()

    device = await discover_esp32()
    # addr = discover()
    logger.info(f"Got Address of ESP32: {device.address}")
    
    client_task = run_ble_client(device, queue)
    consumer_task = run_queue_consumer(queue)

    #try:
    await asyncio.gather(client_task, consumer_task)
    #except DeviceNotFoundError:
    #pass
    
    logger.info("Main method done.")


if __name__ == "__main__":
    asyncio.run(main())
