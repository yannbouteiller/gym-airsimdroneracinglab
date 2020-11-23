# init_rllib_compatibility_server() must be called prior to launching training with the airsim environment

from init_ip_port_and_json_files import init_ip_port_and_json_files
import zmq
from multiprocessing import Process
import time


class Locker():

    def __init__(self, timeout=20.0):
        self.locked = False
        self.last_tick = time.time()
        self.timeout = timeout

    def __iter__(self):
        return self

    def next(self):
        if not self.locked:
            self.locked = True
            self.last_tick = time.time()
            return "go"
        else:
            t = time.time()
            if t - self.last_tick > self.timeout:
                print("DEBUG: LOCKER SERVER TIMEOUT!")
                self.last_tick = time.time()
                return "go"
            return "sorry"


class LockerServer():

    def __init__(self, url="tcp://*:7777"):
        print("Locker server initializing...")
        locker = Locker()
        cnt = zmq.Context()
        sck = cnt.socket(zmq.REP)
        sck.bind(url)
        while True:
            msg = sck.recv_string(flags=0)
            print("DEBUG: server: received:", msg)
            if msg == "acquire":
                sck.send_string(locker.next())
            elif msg == "release":
                print("DEBUG: sever: release request received")
                locker.locked = False
                sck.send_string("released")
            else:
                sck.send_string("unknown request")


def launch_LockerServer():
    LockerServer()


def init_rllib_compatibility_server_and_files(clockspeed=1.0):
    """
    Returns the pid of a LockerServer process (distributed lock)
    """
    print("Initializing locker server...")
    p = Process(target=launch_LockerServer)
    p.start()
    time.sleep(1.0)
    print("Initializing files...")
    init_ip_port_and_json_files(clockspeed=clockspeed)
    print("Locker server ready and airsim files initialized.")
    return p
