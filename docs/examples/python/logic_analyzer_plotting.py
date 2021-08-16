#
# moku example: Basic Logic Analyzer
#
# This example demonstrates how you can configure the Logic
# Analyzer instrument to retrieve a single frame of data for 
# all 16 channels
#
# (c) 2021 Liquid Instruments Pty. Ltd.
#
import matplotlib.pyplot as plt
from moku.instruments import LogicAnalyzer

# Connect to your Moku by its ip address using LogicAnalyzer('192.168.###.###')
# or by its serial number using LogicAnalyzer(serial=123)
i = LogicAnalyzer('10.1.111.201', force_connect=False)

try:
    # Configure the Logic Analyzer pins
    i.set_pins("Pin1", 'O') # Pin 1 as output
    i.set_pins("Pin2", 'H') # Pin 2 as High
    i.set_pins("Pin3", 'L') # Pin 3 as Low
    i.set_pins("Pin4", 'I') # Pin 4 as Input
    i.set_pins("Pin5", 'X') # Pin 5 turned off

    # Print the current state of the instrument
    print(i.summary())

    # Configure the output pattern for Pin 1
    i.generate_pattern("Pin1", [1, 1, 1, 0, 0, 0, 0, 0])

    # Start the output on all pins that are set as output
    i.start_all()

    # Configure the trigger to be Pin 2 with default trigger settings
    i.set_trigger(source='Pin2')

    # Collecte a frame of data from all 16 pins, but only print Pin 1
    data = i.get_data()
    print(data['pin1'])
    

except Exception as e:
    print(f'Exception occurred: {e}')
finally:
    # Close the connection to the Moku device
    # This ensures network resources and released correctly
    i.relinquish_ownership()
