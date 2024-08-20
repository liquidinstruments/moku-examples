#
# moku example: Basic Spectrum Analyzer
#
# This example demonstrates how you can use the Spectrum Analyzer instrument to
# to retrieve a single spectrum data frame over a set frequency span.
#
# (c) 2021 Liquid Instruments Pty. Ltd.
#
from moku.instruments import SpectrumAnalyzer

# Connect to your Moku by its ip address using SpectrumAnalyzer('192.168.###.###')
# or by its serial number using SpectrumAnalyzer(serial=123)
i = SpectrumAnalyzer('192.168.###.###', force_connect=False)

# Deploy the Spectrum Analyzer to your Moku
try:
    # Configure the Spectrum Analyzer 
    i.set_span(0, 10e6)
    i.set_rbw('Auto')  # Auto-mode

    # Get the scan results and print them out (power vs frequency,
    # two channels)
    data = i.get_data()
    print(data['ch1'], data['ch2'], data['frequency'])

except Exception as e:
    print(f'Exception occurred: {e}')
finally:
    # Close the connection to the Moku device
    # This ensures network resources and released correctly
    i.relinquish_ownership()
