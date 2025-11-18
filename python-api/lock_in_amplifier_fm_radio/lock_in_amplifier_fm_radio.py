#
# moku example: FM Radio Receiver with Moku Lock-in Amplifier
#
# This example demonstrates how to configure the Lock-in
# Amplifier instrument to downconvert an input signal from Input 1
# to obtain IQ samples and send them as DIFI packets to a UDP socket.
# This DIFI data can be received and demodulated from GNU Radio or
# similar framework to listen to a FM Radio channel.

# (c) 2024 Liquid Instruments Pty. Ltd.
#

from moku.instruments import LockInAmp
from moku.exceptions import StreamException
import matplotlib.pyplot as plt
import numpy as np
import traceback
import socket
import time

MOKU_IP = '192.168.0.105'
CHANNEL_FREQ = 105.9e6  # Frequnecy of the channel in Hz
CHANNEL_BANDWIDTH = 125e3  # Bandwidth of the channel in Hz
STREAMING_DURATION = None  # Duration of streaming for lock-in amplifier, None for infinite
ENABLE_PLOTTING = False  # Enable plotting
ENABLE_DIFI = True  # Enable sending DIFI
DESTINATION_IP = '127.0.0.1'  # dest address to send packets to
DESTINATION_PORT = 1234  # dest port to send packets to
STREAM_ID = 0  # 32 bit int id of stream
MAX_DATA_LEN = 1017 # max samples per packet

pkt_count = 0 # packet counter for generating seqnum

# Funtion for sending multiple DIFI packets
def send_data(udp_socket, dataI, dataQ):
    for i in range(0, len(dataI), MAX_DATA_LEN):
        send_difi(udp_socket, dataI[i:i+MAX_DATA_LEN], dataQ[i:i+MAX_DATA_LEN])

# Funtion for creating and sending a DIFI packet
def send_difi(udp_socket, dataI, dataQ):
    # 1st 16 bits of header in hex
    pkt_type = 1
    clsid = "1" # 1 bit
    ti = "0" # 1 bits
    vitai = "0" # 1 bit
    vitast = "0" # 1 bit
    tsi = "01" # 2 bits
    tsf = "10" # 2 bits
    packetchunk = int(clsid + ti + vitai + vitast + tsi + tsf, 2)
    global pkt_count
    seqnum = pkt_count % 16
    difi_packet = bytearray.fromhex(f"{pkt_type:01x}{packetchunk:02x}{seqnum:01x}")

    # 2nd 16 bits of header in hex
    difi_packet.extend(bytearray.fromhex(f"{(len(dataI)+7):04x}"))
                                    # number of samples + 7 for the header            
    difi_packet.extend(STREAM_ID.to_bytes(4, 'big')) # add stream id      
    difi_packet.extend(bytearray.fromhex("006A621E")) # XX:006A621E  
                                                      # XX:OUI for Vita       
    difi_packet.extend(bytearray.fromhex("00000000")) # Info Class Code, 
                                                      # Packet Class Code 
                                                      # #icc=0x0000,pcc=0x0000

    # Insert time stamp
    current_time = time.time()
    integer_part = int(current_time) # in seconds
    fractional_part = int((current_time - integer_part) * (1e12)) # in picoseconds
    difi_packet.extend(bytearray.fromhex(f"{integer_part:08x}{fractional_part:016x}"))

    # Insert data
    for i, q in zip(dataI, dataQ):
        difi_packet.extend(bytearray(np.float16([i, q])))

    # Print packet information and send
    # length = len(difi_packet)
    # print(f'Length of this bytes object is {length} bytes and {length/4} 32 bit words')
    udp_socket.sendto(difi_packet, (DESTINATION_IP, DESTINATION_PORT))
    pkt_count = pkt_count % 32 + 1


# Connect to your Moku by its ip address
i = LockInAmp(MOKU_IP, force_connect=True)

try:
    # Set Channel 1 and 2 to DC coupled, 1 MOhm impedance, and
    # 400 mVpp range
    i.set_frontend(1, coupling='AC', impedance='50Ohm',
                   attenuation='0dB')
    i.set_frontend(2, coupling='AC', impedance='50Ohm',
                   attenuation='0dB')

    # Configure the demodulation signal to Local oscillator
    i.set_demodulation('Internal', frequency=CHANNEL_FREQ, phase=0)

    # Set low pass filter to channel bandwidth with 24 dB/octave slope
    i.set_filter(CHANNEL_BANDWIDTH, slope='Slope24dB')

    # Configure output signals
    # X component to Output 1, Y component to Output 2
    i.set_outputs('X', 'Y')

    # Set gain or both outputs to 72dB
    i.set_gain(72,72)

    # Set up signal monitoring
    # Configure monitor points to Input 1 and main output
    i.set_monitor(1, 'MainOutput')
    i.set_monitor(2, 'AuxOutput')

    # Configure the trigger conditions
    # Trigger on Probe A, rising edge, 0V
    i.set_trigger(type='Edge', source='ProbeA', level=0)

    # View +- 1 ms i.e. trigger in the centre
    i.set_timebase(-1e-3, 1e-3)

    i.start_streaming(duration=STREAMING_DURATION, rate=CHANNEL_BANDWIDTH)

    if ENABLE_DIFI:
        # Create the socket
        udp_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    
    if ENABLE_PLOTTING:
        # Set up the plotting parameters
        plt.ion()
        plt.show()
        plt.grid(visible=True)
        plt.ylim([-1, 1])
        line1, = plt.plot([])
        line2, = plt.plot([])
        ax = plt.gca() # Configure labels for axes

    # This loops continuously extracts data and plot/send as configured
    while True:
        # Get new data
        start_time = time.time()
        data = i.get_stream_data()
        if data:
            if ENABLE_DIFI:
                # Send data as DIFI packets
                send_data(udp_socket, data['ch1'], data['ch2'])
            if ENABLE_PLOTTING:
                plt.xlim([data['time'][0], data['time'][-1]])
                # Update the plot
                line1.set_ydata(data['ch1'])
                line2.set_ydata(data['ch2'])
                line1.set_xdata(data['time'])
                line2.set_xdata(data['time'])
                plt.pause(0.001)
        end_time = time.time()
        time_difference = (end_time - start_time) * 10**3
        num_samples = len(data['ch1'])
        print(f'Number of IQ samples: {num_samples}')
        print(f'Time taken for transmit: {time_difference:6.2f} ms')
        print(f'Transmit sample rate: {(len(data['ch1'])/time_difference):6.2f} ksps')
        print("------------------------------------------------------")

except (StreamException,KeyboardInterrupt) as e1: 
    # stop streaming and print message to user
    i.stop_streaming()
    print("Streaming terminated.")

except Exception as e2: 
    # stop streaming and print exception details
    i.stop_streaming()
    print(f'Exception occurred: {e2}')
    print(traceback.format_exc())

finally:
    # Close the connection to the Moku device
    # This ensures network resources are released correctly
    i.relinquish_ownership()
