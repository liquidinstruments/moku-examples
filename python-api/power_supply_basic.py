#
# moku example: Basic Programmable Power Supply Unit
#
#  This example will demonstrate how to configure the power supply
#  units of the Moku:Go.
#
# (c) Liquid Instruments Pty. Ltd.
#
from moku.instruments import Oscilloscope

# Replace with your Moku's IP address.
MOKU_IP = '192.168.###.###'

# Connect to the Moku (force_connect will overtake an existing connection).
# An instrument must be deployed to establish the connection with the
# Moku, in this example we will use the Oscilloscope.
with Oscilloscope(MOKU_IP, force_connect=True) as i:
    # Configure Power Supply Unit 1 to 2 V and 0.1 A
    i.set_power_supply(1, enable=True, voltage=2, current=0.1)

    # Read the current status of Power Supply Unit 1
    print(i.get_power_supply(1))
