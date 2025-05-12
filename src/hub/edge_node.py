import time, asyncio
from bleak import BleakClient, BLEDevice
import logging

from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BLEEdgeClient:
    def __init__(self,
                 device: BLEDevice,
                 queue: asyncio.Queue,
                 esp_uuids: list[str],
                 shutdown_event: asyncio.Event):
        self.device = device
        self.queue = queue
        self.esp_uuids = esp_uuids
        self.shutdown_event = shutdown_event

        # State
        self.addr = device.address
        self.mtu = None
        self.start_conn_time = None
        self.end_conn_time = None

        # Char maps
        self.char_props = {}
        self.notify_uuids = []
        self.read_uuids = []
        self.write_uuids = []

        self.logger = logging.getLogger(__name__)



    async def _callback(self, sender, data):
        await self.queue.put((time.time() - self.start_conn_time, sender, data))

    async def connect(self) -> bool:
        """Connect, discover chars, and subscribe to notifies."""
        self.client = BleakClient(self.device)
        await self.client.connect()
        if not await self.client.is_connected():
            return False

        self.mtu            = self.client.mtu_size
        self.start_conn_time = time.time()

        # Discover and catalog
        services = await self.client.get_services()
        for svc in services:
            for ch in svc.characteristics:
                uuid, props = ch.uuid, ch.properties
                self.char_props[uuid] = props
                if "notify" in props:
                    self.notify_uuids.append(uuid)
                if "read" in props:
                    self.read_uuids.append(uuid)
                if "write" in props or "write-without-response" in props:
                    self.write_uuids.append(uuid)

        # Subscribe
        for uuid in self.notify_uuids:
            if uuid in self.esp_uuids:
                await self.client.start_notify(uuid, self._callback)
                self.logger.info(f"Started notify with UUID: {uuid}")

        return True

    async def disconnect(self):
        """Stop notifies and close connection."""
        # stop notifications
        for uuid in self.notify_uuids:
            await self.client.stop_notify(uuid)

        self.end_conn_time = time.time()
        # signal end-of-data
        await self.queue.put((self.end_conn_time, None, None))

        # finally disconnect the BleakClient
        await self.client.disconnect()


    # def packet_ready():
    #     if

    async def heartbeat_loop(
        self,
        interval: float = 30.0,
        max_retries: int = 3,
        retry_delay: int = 5
    ):
        try:
            while True:
                alive = await self.client.is_connected
                if not alive:
                    self.logger.warning(f"Heartbeat: {self.addr} disconnected; reconnecting…")
                    for attempt in range(1, max_retries + 1):
                        try:
                            await self.client.connect()
                            if await self.client.is_connected:
                                self.logger.info(f"Reconnected to {self.addr}")
                                break
                            else:
                                self.logger.error(
                                    f"Reconnect attempt {attempt} failed (still disconnected)"
                                )
                        except Exception as e:
                            self.logger.error(f"Reconnect attempt {attempt} error: {e}")

                        if attempt < max_retries:
                            await asyncio.sleep(retry_delay)
                        else:
                            self.logger.error(
                                f"Failed to reconnect after {max_retries} attempts."
                            )
                else:
                    self.logger.debug(f"Heartbeat OK: {self.addr}")
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            self.logger.info("Heartbeat task cancelled")
            raise
