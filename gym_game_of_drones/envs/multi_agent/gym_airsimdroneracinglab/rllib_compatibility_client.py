import zmq
import time


class LockerClient():

    def __init__(self, url="tcp://localhost:7777"):
        cnt = zmq.Context()
        self.sck = cnt.socket(zmq.REQ)
        self.sck.connect(url)

    def next(self):
        self.sck.send_string("acquire")
        ret = self.sck.recv_string(flags=0)
        if ret == "go":
            return True
        else:
            print("DEBUG: received", ret)
            return False

    def acquire(self, waiting_time_between_requests=20.0):
        print("DEBUG: LockerClient sends acquire request")
        acquired = self.next()
        while acquired is False:
            time.sleep(waiting_time_between_requests)
            acquired = self.next()
        print("DEBUG: LockerClient acquired")

    def release(self):
        self.sck.send_string("release")
        msg = ""
        while msg != "released":
            msg = self.sck.recv_string(flags=0)
            if msg == "released":
                print("DEBUG: LockerClient released")
            else:
                print("DEBUG: WARNING (release): BAD MESSAGE RECEIVED")
