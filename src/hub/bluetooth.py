import time
import bluetooth

try:
    from bluetooth.ble import DiscoveryService
except ImportError:
    DiscoveryService = None

def discover():

    """
    currently Linux only
    """
    if DiscoveryService is None:
        return None

    svc = DiscoveryService()
    dev_ble:dict = svc.discover(10)

    if dev_ble:
        # for u, n in dev_ble.items():
        # if "XIAOEXP32S3_BLE_SERVER" in dev_ble.keys():
        print(f"Dev BLE: {dev_ble}")

        for u, n in dev_ble.items():
            if n == "XIAOEXP32S3_BLE_SERVER":
                return u


            

def connect(ble_addr):
    while(True):
        try:
            BT_Mate = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            BT_Mate.connect((str(ble_addr), 1))
            break
        except bluetooth.btcommon.BluetoothError as error:
            BT_Mate.close()
            print ("Could not connect: ", error, "; Retrying in 10 secs")
            time.sleep(10)
    return BT_Mate

if __name__ == "__main__":

    addr = discover()

    BT = connect(addr)

    while(True):
        data = BT.recv(1024)
        print(data)