#
# moku example: Basic Spectrum Analyzer
#
# This example demonstrates how you can use the Spectrum Analyzer instrument to
# to retrieve a single spectrum data frame over a set frequency span.
#
# (c) Liquid Instruments Pty. Ltd.
#
from moku.instruments import SpectrumAnalyzer

# Replace with your Moku's IP address.
MOKU_IP = '192.168.###.###'

# Connect to the Moku (force_connect will overtake an existing connection).
with SpectrumAnalyzer(MOKU_IP, force_connect=True) as i:
    # Deploy the Spectrum Analyzer to your Moku
    # Configure the Spectrum Analyzer
    i.set_span(0, 10e6)
    i.set_rbw('Auto')  # Auto-mode

    # Get the scan results and print them out (power vs frequency,
    # two channels)
    data = i.get_data()
    print(data['ch1'], data['ch2'], data['frequency'])
