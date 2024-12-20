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

MOKU_IP = '192.168.0.109'
CHANNEL_FREQ = 105.9e6
STREAMING_DURATION = 50
ENABLE_PLOTTING = False # Enable plotting
ENABLE_DIFI = True # Enable sending DIFI
DESTINATION_IP = '127.0.0.1'  # dest address to send packets to
DESTINATION_PORT = 1234  # dest port to send packets to
STREAM_ID = 0  # 32 bit int id of stream
MAX_DATA_LEN = 1017 # max samples per packet

pkt_count = 0 # packet counter for generating seqnum

# Funtion for sending multiple DIFI packets
def send_data(udp_socket, dataI, dataQ, lenData):
    for i in range(0,lenData, MAX_DATA_LEN):
        curPos = i
        curLen = MAX_DATA_LEN   # default samples per packet
        if lenData < MAX_DATA_LEN:
            curLen = lenData    # adjust the number of samples if 
                                # the size of passed array is 
                                # less than the default
        if curPos + curLen > lenData:
            curLen = lenData - curPos # set number of samples 
                                      # for the last packet
        send_difi(udp_socket, dataI[curPos:curPos+curLen], 
                  dataQ[curPos:curPos+curLen], curLen) # send a DIFI packet

# Funtion for creating and sending a DIFI packet
def send_difi(udp_socket, dataI, dataQ, lenData):
    # 1st 16 bits of header in hex
    pkt_type = "1"
    clsid = "1" # 1 bit
    rsvd = "00" # 2 bits
    tsm = "0" # 1 bit
    tsi = "01" # 2 bits
    tsf = "10" # 2 bits
    clsid_rsvd_tsm_tsi_tsf_binary = clsid + rsvd + tsm + tsi + tsf # 10000110
    clsid_rsvd_tsm_tsi_tsf_dec = int(clsid_rsvd_tsm_tsi_tsf_binary, 2) # dec
    clsid_rsvd_tsm_tsi_tsf = "%02x" % clsid_rsvd_tsm_tsi_tsf_dec # hex
    global pkt_count
    seqnum = "%01x" % (pkt_count % 16)
    first_half_header = "%s%s%s" % (pkt_type, clsid_rsvd_tsm_tsi_tsf, seqnum)
    packetchunk = bytearray.fromhex(first_half_header) # (1A61) 
                                                       # clsid=0x1,
                                                       # tsm=0x0,tsf=0x2
    
    difi_packet = bytearray() # prep vita/difi payload
    difi_packet.extend(packetchunk)

    # 2nd 16 bits of header in hex, (Packet Size)
    # number of samples + 7 for the header
    difi_packet.extend(bytearray.fromhex(hex(lenData+7)[2:].zfill(4)))                    
    difi_packet.extend(STREAM_ID.to_bytes(4, 'big')) # add stream id      
    difi_packet.extend(bytearray.fromhex("006A621E")) # XX:006A621E  
                                                      # XX:OUI for Vita       
    difi_packet.extend(bytearray.fromhex("00000000")) # Info Class Code, 
                                                      # Packet Class Code 
                                                      # #icc=0x0000,pcc=0x0000

    # Insert time stamp
    packet_timestamp = format(int(time.time()),'x') # Integer part
    difi_packet.extend(bytearray.fromhex(packet_timestamp))
    difi_packet.extend(bytearray.fromhex("0000000000000001")) # Fractional part

    # Insert data
    for i in range(0, lenData): # fill data to the full length of packet
        difi_packet.extend(bytearray(np.float16(dataI[i])))
        difi_packet.extend(bytearray(np.float16(dataQ[i])))

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

    # Set low pass filter to 200 kHz corner frequency with 24 dB/octave slope
    i.set_filter(0.2e6, slope='Slope24dB')

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

    i.start_streaming(duration=STREAMING_DURATION, rate=125e3)

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
                send_data(udp_socket, data['ch1'], 
                          data['ch2'], len(data['ch1']))
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
        print("Number of IQ samples: ", num_samples)
        print("Time taken for transmit: ", time_difference, "ms")
        print("Transmit sample rate: ",
              num_samples/time_difference, "ksps")
        print("-------------------------------------------------------")

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
