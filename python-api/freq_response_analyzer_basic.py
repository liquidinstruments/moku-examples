#
# moku example: Basic Frequency Response Analyzer
#
# This example demonstrates how you can generate output sweeps using the
# Frequency Response Analyzer instrument, and view one frame of the transfer
# function data.
#
# (c) Liquid Instruments Pty. Ltd.
#
from moku.instruments import FrequencyResponseAnalyzer

# Replace with your Moku's IP address.
DEVICE_IP = '192.168.###.###'

# Connect to the Moku (force_connect will overtake an existing connection).
with FrequencyResponseAnalyzer(DEVICE_IP, force_connect=True) as i:
    # Configure output sweep parameters (100Hz-20MHz)
    i.set_sweep(
        start_frequency=20e6,
        stop_frequency=100,
        num_points=256,
        averaging_time=1e-3,
        averaging_cycles=5,
        settling_cycles=5,
        settling_time=1e-3,
    )

    # Configure output sweep amplitudes
    # Channel 1 - 0.1Vpp
    # Channel 1 - 0.1Vpp
    i.set_output(1, 0.1)
    i.set_output(2, 0.1)

    # Start the sweep
    i.start_sweep()

    # Get a single sweep frame. This will block until the sweep is complete,
    # beware if your range includes low frequencies!
    frame = i.get_data(wait_complete=True)

    # Print out the data for Channel 1
    print(frame['ch1']['frequency'], frame['ch1']['magnitude'], frame['ch1']['phase'])

    # Print out the data for Channel 2
    print(frame['ch2']['frequency'], frame['ch2']['magnitude'], frame['ch2']['phase'])
