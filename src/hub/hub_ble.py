
from bleak import BleakScanner, BleakClient
import asyncio
import time
import logging
import struct

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


logger = logging.getLogger(__name__)

AUIDO_CHAR_UUID = "00002A57-0000-1000-8000-00805F9B34FB"


async def discover_edge_devices(esp_name: str= "XIAOESP32S3_BLE_SERVER", max_devices:int=2):

    devices = []
    discover_dict = await BleakScanner.discover(adapter="hci1", return_adv=True)
    while len(devices) == 0:
        # dev_dict[0]: BLEDevice, dev_dict[1]: AdvertisementData
        for dev_addr, dev_dict in discover_dict.items():
            # logger.info(f"U: {dev_addr}, N: {dev_dict[0].name}, rssi: {dev_dict[0].rssi}")
            
            if dev_dict[0].name == esp_name:
                logger.info(f"found ESP32 at addr: {dev_addr}, UUIDs: {dev_dict[1]}")

                if len(devices) < max_devices:
                    devices.append(dev_dict[0]) # For Bleak better to connect with BLEDevice class
                    break
                else:
                    logger.info(f"Max Device connection exceeded, not adding device: {dev_addr}")
            else:
                logger.info(f"Target device not found. Retrying after 1 second...")
                await asyncio.sleep(1)

    return devices
"""
async def run_queue_consumer(queue: asyncio.Queue):
    logger.info("Starting queue consumer")
 
    init_recv_t = 0

    client_recv_data = {}
    while True:
        # Use await asyncio.wait_for(queue.get(), timeout=1.0) if you want a timeout for getting data.
        try:
            epoch, sender, data = await asyncio.wait_for(queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        
        if data is None:
            logger.info(
                "Got message from client about disconnection. Exiting consumer loop..."
            )
            break
        else:
 
 
            # Byte Array sent from ESP32
            data = struct.unpack('<' + 'H' *(len(data) // 2), data)
 
            logger.info("Received callback data via async queue at %s: %r, from %s", epoch, data, sender)
 
 
            if str(sender) not in client_recv_data:
                client_recv_data[str(sender)] = []
            client_recv_data[str(sender)].extend(data)
 
            with open(f"sender_data.txt", "a") as f:
                f.write(str(client_recv_data[str(sender)]) + "\n\n")
                f.flush()
 
            # if 1 in data:
            #     init_recv_t = time.time()
 
            # if 1000 in data:
            #     logger.info(f"Recv 1000 Samples after {time.time() - init_recv_t} s")
"""
async def run_ble_client(device, queue:asyncio.Queue):
    
    start_notify_time = 0

    async def callback_handler(sender, data):
        
        await queue.put((time.time() - start_notify_time, sender, data))

    async with BleakClient(device) as client:
        
        if not await client.is_connected():
            logger.error(f"Failed to Connect")
            return

        logger.info(f" Connected to Device: {device.address}")
        
        # client.set_mtu_size(517)
        logger.info(f" MTU size: {client.mtu_size}")

        services = await client.get_services()
        
        read_char = None
        for service in services:
            logger.info(f"Service: {service.uuid}")
            for characteristic in service.characteristics:
                logger.info(f"  Characteristic: {characteristic.uuid}")
                logger.info(f"    Properties: {characteristic.properties}")

                if (characteristic.uuid == AUIDO_CHAR_UUID) or ('notify' in characteristic.properties): # Is audio Array characteristic
                    logger.info(f"Characteristic uuid: {characteristic.uuid}")
                    await client.start_notify(characteristic.uuid, callback_handler)
                    start_notify_time = time.time()
                    await asyncio.sleep(60)

                    await client.stop_notify(characteristic.uuid)
                    """
                    Need to convert array to s16le .wav file with 16kHz sampling rate
                    """
                    
                if 'notify' in characteristic.properties:
                    value = await client.read_gatt_char(characteristic.uuid)
                    logger.info(f"   Value: {value}")
                    read_char = characteristic.uuid

        # await client.start_notify(read_char, callback_handler)
        # await asyncio.sleep(30.0)

        # await client.stop_notify(read_char)

        await queue.put((time.time(), None, None))

        ble_check = asyncio.create_task(heartbeat_loop(client, 15.0))
        logger.info(f"Queue: {queue._queue}")

        #clean up heartbeat (not sure if we need this...)
        ble_check.cancel()
        try:
            await ble_check
        except asyncio.CancelledError:
            pass
        logger.info("Heartbeat loop terminated")
        
async def heartbeat_loop(
    client, 
    interval: float = 30.0,
    max_retries: int = 3,
    retry_delay: int = 5):
    try:
        while True:
            alive = await client.is_connected()
            if not alive:
                logger.warning(f"Heartbeat: {client.address} disconnected; reconnecting…")
                for attempt in range(1, max_retries + 1):
                    try:
                        await client.connect()
                        if await client.is_connected():
                            logger.info(f"Reconnected to {client.address}")
                            break
                        else:
                            logger.error(f"Reconnect attempt {attempt} failed (still disconnected)")

                    except Exception as e:
                        logger.error(f"Reconnect attempt {attempt} raised error: {e}")
                    
                    if attempt < max_retries:
                        logger.debug(f"Waiting {retry_delay}s before next reconnect attempt…")
                        await asyncio.sleep(retry_delay)
                    else:
                        logger.error(f"Failed to reconnect after {max_retries} attempts.")
            else:
                logger.debug(f"Heartbeat OK: {client.address}")
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        logger.info("Heartbeat task cancelled")
        raise

async def main():

    init_recv = 0
    queue = asyncio.Queue()

    device = await discover_edge_devices()
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
