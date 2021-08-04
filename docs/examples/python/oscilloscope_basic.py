#
# moku example: Basic Oscilloscope
#
# This script demonstrates how to use the Oscilloscope instrument
# to retrieve a single frame of dual-channel voltage data.
#
# (c) 2021 Liquid Instruments Pty. Ltd.
#
from moku.instruments import Oscilloscope

# Connect to your Moku by its ip address using Oscilloscope('192.168.###.###')
# or by its serial number using Oscilloscope(serial=123)
i = Oscilloscope('10.1.111.201', force_connect=True)
# print(i.describe())
print(i.available_power_supplies())
i.list()
i.upload()
i.download()
i.delete()
# i.read_power_supply()
# i.set_power_supply()
exit(0)
try:
    # Span from -1s to 1s i.e. trigger point centred
    i.set_timebase(-1, 1)

    # Get and print a single frame worth of data (time series
    # of voltage per channel)
    data = i.get_data()
    print(data['ch1'], data['ch2'], data['time'])

except Exception as e:
    print(f'Exception occurred: {e}')
finally:
	    # Close the connection to the Moku device
    # This ensures network resources and released correctly
    i.relinquish_ownership()
