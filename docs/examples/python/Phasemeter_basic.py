# moku example: Basic Phasemeter
#
# This script demonstrates how to use the Phasemeter instrument
# to retrieve data from 4 channels.
#
# (c) 2021 Liquid Instruments Pty. Ltd.
#
from moku.instruments import Phasemeter

# Connect to your Moku by its ip address using Phasemeter('192.168.###.###')
# or by its serial number using Phasemeter(serial=123)
i = Phasemeter('192.168.###.###', force_connect=False)

try:
    # Sets the acquisition speed to 480 Hz
    i.set_acquisition_speed(speed='480Hz')

    # Configure the Phasemeter loops
    # All channels to 30 MHz, bandwidth of 2.5 kHz
    i.set_pm_loop(channel=1, auto_acquire=False, frequency=30e6, bandwidth="2k5Hz")
    i.set_pm_loop(channel=2, auto_acquire=False, frequency=30e6, bandwidth="2k5Hz")
    i.set_pm_loop(channel=3, auto_acquire=False, frequency=30e6, bandwidth="2k5Hz")
    i.set_pm_loop(channel=4, auto_acquire=False, frequency=30e6, bandwidth="2k5Hz")

    # Configure the frontend of each channel
    i.set_frontend(channel=1, impedance='50Ohm', coupling='DC', range='4Vpp')
    i.set_frontend(channel=2, impedance='50Ohm', coupling='DC', range='4Vpp')
    i.set_frontend(channel=3, impedance='50Ohm', coupling='DC', range='4Vpp')
    i.set_frontend(channel=4, impedance='50Ohm', coupling='DC', range='4Vpp')
    
    # Get and print a single frame worth of data 
    data = i.get_data()
    # Print out the measured phase of each channel
    print(data['ch1']['phase'])
    print(data['ch2']['phase'])
    print(data['ch3']['phase'])
    print(data['ch4']['phase'])

except Exception as e:
    print(f'Exception occurred: {e}')
finally:
    # Close the connection to the Moku device
    # This ensures network resources and released correctly
    i.relinquish_ownership()
