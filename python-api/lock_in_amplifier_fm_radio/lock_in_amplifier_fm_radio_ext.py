from moku.exceptions import StreamException
from moku.instruments import LockInAmp
import numpy as np
from threading import Thread
import socket
import time
import traceback
import tty
import termios
import sys

MOKU_IP = '192.168.0.105'
CHANNEL_FREQ = 105.9e6  # Frequnecy of the channel in Hz
CHANNEL_BANDWIDTH = 125e3  # Bandwidth of the channel in Hz
CHANNEL_SPACING = 800e3  # Channel spacing in Hz
CHANNEL_CHANGE_OPTION = 1  # 1 for re-instantiating the stream (works),
                            # 2 for restarting the stream (doesn't work)
STREAMING_DURATION = 20  # Duration of streaming for lock-in amplifier
ENABLE_PLOTTING = False  # Enable plotting
DESTINATION_IP = '127.0.0.1'  # dest address to send packets to
DESTINATION_PORT = 1234  # dest port to send packets to
STREAM_ID = 0  # 32 bit int id of stream
MAX_DATA_LEN = 1017 # max samples per packet

class grmoku(object):

    def __init__(self, moku_ip, chnl_freq, chnl_bw, duration,
                 dest_ip, dest_port, stream_id, max_samples):
        self.moku_ip = moku_ip
        self.chnl_freq = chnl_freq
        self.chnl_bw = chnl_bw
        self.duration = duration
        self.dest_ip = dest_ip
        self.dest_port = dest_port
        self.stream_id = stream_id
        self.max_samples = max_samples
        self.pkt_count = 0
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  
        self.chnl_freq_changed = False 
        self.run_thread = True
        self.thread = Thread(target=self.stream)
        self.thread.start()

    def __del__(self):
        self.run_thread = False
        self.thread.join()
        self.udp_socket.close()

    def stream(self):
        try:
            self.instrument = LockInAmp(self.moku_ip, force_connect=True)
            self.instrument.set_frontend(1, coupling='AC', impedance='50Ohm',
                    attenuation='0dB')
            self.instrument.set_frontend(2, coupling='AC', impedance='50Ohm',
                    attenuation='0dB')
            self.instrument.set_demodulation('Internal', frequency=self.chnl_freq, phase=0)
            self.instrument.set_filter(200e3, slope='Slope24dB')
            self.instrument.set_outputs('X', 'Y')
            self.instrument.set_gain(72,72)
            self.instrument.set_monitor(1, 'MainOutput')
            self.instrument.set_monitor(2, 'AuxOutput')
            self.instrument.set_trigger(type='Edge', source='ProbeA', level=0)
            self.instrument.set_timebase(-1e-3, 1e-3)
            self.instrument.start_streaming(duration=self.duration, rate=self.chnl_bw)
            print('Start streaming...')
            print('Press arrow keys to change channel frequency, other keys to exit')
            while self.run_thread:
                if self.chnl_freq_changed:
                    self.instrument.stop_streaming()
                    self.instrument.set_demodulation('Internal', 
                                    frequency=self.chnl_freq, phase=0)
                    self.chnl_freq_changed = False
                    self.instrument.start_streaming(duration=self.duration,
                                                    rate=self.chnl_bw)
                start_time = time.time()
                data = self.instrument.get_stream_data()
                if data:
                    self.send_data(data['ch1'], data['ch2'])
                end_time = time.time()
                time_difference = (end_time - start_time) * 10**3
                num_samples = len(data['ch1'])
                print(f'Number of IQ samples: {num_samples}')
                print(f'Time taken for transmit: {time_difference:6.2f} ms')
                print(f'Transmit sample rate: {(num_samples/time_difference):6.2f} ksps')
                print("------------------------------------------------------")

        except Exception as e:           
            print(f'Exception occurred: {e}')
            print(traceback.format_exc())
        finally:
            self.instrument.stop_streaming()
            self.instrument.relinquish_ownership()
            print("Streaming terminated")
    
    def send_data(self, dataI, dataQ):
        for i in range(0, len(dataI), self.max_samples):
            self.send_difi(dataI[i:i+self.max_samples], dataQ[i:i+self.max_samples])

    def send_difi(self, dataI, dataQ):
        # 1st 16 bits of header in hex
        pkt_type = 1
        clsid = "1" # 1 bit
        ti = "0" # 1 bits
        vitai = "0" # 1 bit
        vitast = "0" # 1 bit
        tsi = "01" # 2 bits
        tsf = "10" # 2 bits
        packetchunk = int(clsid + ti + vitai + vitast + tsi + tsf, 2)
        seqnum = self.pkt_count % 16
        difi_packet = bytearray.fromhex(f"{pkt_type:01x}{packetchunk:02x}{seqnum:01x}")

        # 2nd 16 bits of header in hex
        difi_packet.extend(bytearray.fromhex(f"{(len(dataI)+7):04x}"))
                                        # number of samples + 7 for the header            
        difi_packet.extend(self.stream_id.to_bytes(4, 'big')) # add stream id      
        difi_packet.extend(bytearray.fromhex("006A621E")) # XX:006A621E  
                                                        # XX:OUI for Vita       
        difi_packet.extend(bytearray.fromhex("00000000")) # Info Class Code, 
                                                        # Packet Class Code 
                                                        # #icc=0x0000,pcc=0x0000

        # Insert time stamp
        current_time = time.time()
        integer_part = int(current_time)
        fractional_part = int((current_time - integer_part) * (1e12))
        difi_packet.extend(bytearray.fromhex(f"{integer_part:08x}{fractional_part:016x}"))
        
        # Insert data
        for i, q in zip(dataI, dataQ):
            difi_packet.extend(bytearray(np.float16([i, q])))

        # Print packet information and send
        # length = len(difi_packet)
        # print(f'Length of this bytes object is {length} bytes and {length/4} 32 bit words')
        self.udp_socket.sendto(difi_packet, (self.dest_ip, self.dest_port))
        self.pkt_count = self.pkt_count % 32 + 1
    
    def set_chnl_freq(self, increament, option = 1):
        if option == 1:
            self.chnl_freq += increament
            print(f'Opt1 setting channel frequency to {self.chnl_freq} Hz')
            self.run_thread = False
            self.thread.join()
            self.run_thread = True
            self.thread = Thread(target=self.stream)
            self.thread.start()
        elif option == 2:
            self.chnl_freq += increament
            print(f'Opt2 setting channel frequency to {self.chnl_freq} Hz')
            self.chnl_freq_changed = True

def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(3)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

grmoku_object = grmoku(MOKU_IP, CHANNEL_FREQ, CHANNEL_BANDWIDTH,
                STREAMING_DURATION, DESTINATION_IP, DESTINATION_PORT,
                STREAM_ID, MAX_DATA_LEN)
while True:
    key = get_key()
    if key == '\x1b[A' or key == '\x1b[C':  # Up or right arrow key
        grmoku_object.set_chnl_freq(CHANNEL_SPACING, CHANNEL_CHANGE_OPTION)
    elif key == '\x1b[B' or key == '\x1b[D':  # Down or left arrow key
        grmoku_object.set_chnl_freq(-CHANNEL_SPACING, CHANNEL_CHANGE_OPTION)
    else:
        print('Request to exit')
        grmoku_object.__del__()
        break
sys.exit(0)