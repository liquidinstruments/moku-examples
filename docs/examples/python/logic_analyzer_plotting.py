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
    print(i.get_pins())
    # Configure the Logic Analyzer pins
    i.set_pins("Pin1", 'O') # Pin 1 as output
    i.set_pins("Pin2", 'H') # Pin 2 as High
    i.set_pins("Pin3", 'L') # Pin 3 as Low
    i.set_pins("Pin4", 'I') # Pin 4 as Input
    i.set_pins("Pin5", 'X') # Pin 5 turned off

    # Configure the output pattern for Pin 1
    i.generate_pattern("Pin1", [1, 0, 0, 0, 0, 0, 0, 0])

    # Start the output on all pins that are set as output
    i.start_all()

    # Collecte a frame of data from all 16 pins
    # data = i.get_data()
    # print(data['pin1'])
    print(i.summary())
except Exception as e:
    raise e
    # print(f'Exception occurred: {e}')

finally:
    # Close the connection to the Moku device
    # This ensures network resources and released correctly
    i.relinquish_ownership()
