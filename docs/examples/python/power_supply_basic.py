#
# moku example: Basic Programmable Power Supply Unit 
#
#  This example will demonstrate how to configure the power supply
#  units of the Moku:Go.
#
# (c) 2021 Liquid Instruments Pty. Ltd.
#
from moku.instruments import Datalogger

# Connect to your Moku by its ip address using Datalogger('192.168.###.###')
# or by its serial number using Datalogger(serial=123)
# An instrument must be deployed to establish the connection with the
# Moku, in this example we will use the Datalogger.
i = Datalogger('192.168.###.###', force_connect=False)

try:
    # Configure Power Supply Unit 1 to 2 V and 0.1 A
    i.set_power_supply(1,enable=True,voltage=2, current=0.1)
    
    # Read the current status of Power Supply Unit 1 
    print(i.read_power_supply(1))

except Exception as e:
    print(f'Exception occurred: {e}')
finally:
    # Close the connection to the Moku device
    # This ensures network resources and released correctly
    i.relinquish_ownership()
