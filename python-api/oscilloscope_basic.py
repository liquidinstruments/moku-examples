#
# moku example: Basic Oscilloscope
#
# This script demonstrates how to use the Oscilloscope instrument
# to retrieve a single frame of dual-channel voltage data.
#
# (c) Liquid Instruments Pty. Ltd.
#
from moku.instruments import Oscilloscope

# Replace with your Moku's IP address.
MOKU_IP = '192.168.###.###'

# Connect to the Moku (force_connect will overtake an existing connection).
with Oscilloscope(MOKU_IP, force_connect=True) as i:
    # Set the span to from -1ms to 1ms i.e. trigger point centred
    i.set_timebase(-1e-3, 1e-3)

    # Get and print a single frame  of data (time series
    # of voltage per channel)
    data = i.get_data()
    print(data['ch1'], data['ch2'], data['time'])
