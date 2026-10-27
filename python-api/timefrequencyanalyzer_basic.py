#
# moku example: Plotting Time and Frequency Analyzer
#
# This example demonstrates how you can configure the Time and Frequency Analyzer
# instrument, and view the statistics of the intervals.
#
# (c) Liquid Instruments Pty. Ltd.
#
from pprint import pprint

from moku.instruments import TimeFrequencyAnalyzer

# Replace with your Moku's IP address.
MOKU_IP = '192.168.###.###'

# Connect to the Moku (force_connect will overtake an existing connection).
with TimeFrequencyAnalyzer(MOKU_IP, force_connect=True) as i:
    # Set the event detectors
    # Set Event A to detect rising edge events on Input 1 at 0V
    # Set Event B to detect rising edge events on Input 2 at 0V
    i.set_event_detector(1, 'Input1', threshold=0, edge='Rising')
    i.set_event_detector(2, 'Input2', threshold=0, edge='Rising')

    # Set the interpolation to Linear
    i.set_interpolation(mode='Linear')

    # Set acquisition mode to a 100ms Window
    i.set_acquisition_mode(mode='Windowed', window_length=100e-3)

    # Use the first start event and close the interval at the end of the gate period (arbitrary as using windowed acquisition)
    i.set_interval_policy(multiple_start_events='UseFirst', incomplete_intervals='Close')

    # Set the interval analyzers
    # Set Interval A to start at Event A and stop at Event A
    # Set Interval B to start at Event B and stop at Event B
    i.set_interval_analyzer(1, start_event_id=1, stop_event_id=1)
    i.set_interval_analyzer(2, start_event_id=2, stop_event_id=2)

    # Get data and explore statistics
    data = i.get_data()
    print('Interval 1:')
    pprint(data['interval1']['statistics'])
    print('Interval 2:')
    pprint(data['interval2']['statistics'])
