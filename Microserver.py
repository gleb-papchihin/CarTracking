
from threading import Thread
import threading
import socket
import json
import cv2


class Microserver:
    
    def __init__(self, ip, port, thread, listen=1, buffer=1024):
        
        assert isinstance(ip,   str), 'IP must be a string.'
        assert isinstance(port, int), 'PORT must be a int.'
        assert 1024 < port < 65536,   'PORT must be in (1024, 65536) interval.'
        
        self.ip     = ip
        self.port   = port
        self.alive  = True
        self.buffer = buffer
        self.thread = thread
        self.sock   = self.create_socket(ip, port, listen)
        
        self.start()
    
    
    def create_socket(self, ip, port, listen=1):
        family      = socket.AF_INET     # IPv4
        socket_type = socket.SOCK_STREAM # TCP
        sock        = socket.socket(family, socket_type)
        sock.bind((ip, port))
        sock.listen(listen)
        return sock
    
    
    def create_init_message(self):
        message = {
            'ip'     : self.ip,
            'port'   : self.port,
            'alive'  : self.alive,
            'message': 'Be my guest :)'
        }
        
        message = json.dumps(message)
        message = bytes(message, 'utf8')
        return message
    
    
    def create_message(self, collection):
        message = json.dumps(collection)
        message = bytes(message, 'utf8')
        return message
    
    
    def read_request(self, request):
        request = request.decode('utf8')
        request = json.loads(request)
        return request

    
    def start(self):
        
        while self.alive:
            
            try:
                clientsocket, address = self.sock.accept()
            except:
                continue
            
            if address[0] != '127.0.0.1':
                clientsocket.close()
                continue
            
            try:
                message = self.create_init_message()
                clientsocket.send(message)
            
                request = clientsocket.recv(self.buffer)
                request = self.read_request(request)
                
                command = request.get('command')
                command = int(command)
                
                if command == 1:
                    current = self.thread.current
                    message = self.create_message({'current': current})
                    clientsocket.send(message)

                elif command == 2:
                    message = self.create_message({'alive': False})
                    clientsocket.send(message)
                    clientsocket.close()
                    self.stop()
                    
                else:
                    message = self.create_message({'Error': 'Unknown command'})
                    clientsocket.send(message)
                
                clientsocket.close()
            except:
                clientsocket.close()
                continue
        
        
    def stop(self):
        self.alive = False
        self.thread.stop()
        self.sock.close()
    
    def __exit__(self, type, value, traceback):
        self.alive = False
        self.thread.stop()
        self.sock.close()

class AsyncCapture(Thread):
    
    def __init__(self, rtsp):
        
        assert isinstance(rtsp, str), 'RTSP must be a string'
        assert rtsp != '0',           'Incorrect RTSP address'
        
        Thread.__init__(self, name='Capture')
        
        self.capture = cv2.VideoCapture(rtsp)
        self.lock    = threading.Lock()
        self.grab    = False
        self.frame   = None
        self.alive   = True
    
    def run(self):
        while self.alive:
            grab, frame = self.capture.read()
            
            with self.lock:
                self.grab  = grab
                self.frame = frame
    
    def read(self):
        with self.lock:
            grab  = self.grab
            frame = self.frame
        return grab, frame
    
    def isOpened(self):
        return self.capture.isOpened()
    
    def stop(self):
        self.alive = False
    
    def __exit__(self, type, value, traceback):
        self.alive = False
        self.capture.release()

class Stream(Thread):
    
    def __init__(self, rtsp, points, weight, cfg):
        
        Thread.__init__(self, name='Stream')
        
        self.detector  = Detector(weight, cfg)
        self.transform = Transform(points)
        self.alive     = True
        self.current   = 0
        
        self.capture   = AsyncCapture(rtsp)
        self.capture.start()
        
    def run(self):
        while self.alive:
            grab, frame = self.capture.read()
            self.current += 1
            time.sleep(1)
            
    def stop(self):
        self.alive = False
        self.capture.stop()
    
    def __exit__(self):
        self.alive = False
        self.capture.stop()


