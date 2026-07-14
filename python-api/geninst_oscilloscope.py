#
# moku example: GenInst
#
# This example demonstrates how you can configure GenInst using
# Multi-Instrument Mode and output the result to the Oscilloscope.
# This example is written for Moku:Pro and can be changed for
# other devices.
#  (c) Liquid Instruments Pty Ltd
#

import matplotlib.pyplot as plt
from moku.instruments import GenInst, MultiInstrument, Oscilloscope
# Connect to your Moku
# Connect to Moku via its IP address. Change platform_id to:
#   - 3, 5 or 8 for Moku:Delta, or
#   - 4 for Moku:Pro, or
#   - 2 or 3 for Moku:Lab and Moku:Go.
# force_connect will overtake an existing connection

m = MultiInstrument('192.168.###.###', platform_id=4, force_connect=True)

try:
    # Set the instruments and upload GenInst bitstreams from your device
    # to your Moku
    bitstream = "path/to/geninst/bitstreams"
    gi = m.set_instrument(1, GenInst, bitstream=bitstream)
    osc = m.set_instrument(2, Oscilloscope)

    # Configure the connections
    connections = [
        dict(source="Slot1OutA", destination="Slot2InA"),
        dict(source="Slot1OutB", destination="Slot2InB"),
        dict(source="Slot2OutA", destination="Slot1InA"),
        dict(source="Slot2OutB", destination="Slot1InB"),
    ]
    m.set_connections(connections=connections)

    # Set Control Registers as per the GenInst design plan
    # Setting a single control register
    gi.set_control(1, 0b00)

    #  Setting multiple control registers
    controls = [
    {"idx": 0, "value": 32836},
    {"idx": 1, "value": 450},
    {"idx": 2, "value": 32},
    {"idx": 3, "value": 2048},
    {"idx": 4, "value": 2147450879}
    ]
    
    gi.set_controls(controls) 
    
    # Configure the Oscilloscope to generate different waveforms
    osc.generate_waveform(1, 'Sine', amplitude=1, frequency=1e3)
    osc.generate_waveform(2, 'Sine', amplitude=0.1, frequency=500)

    # Sync the phase between the waveforms
    osc.sync_output_phase()

    # Set the time span to cover four cycles of the waveforms
    osc.set_timebase(-2e-3, 2e-3)

    # Get initial data frame to set up plotting parameters. This can be done
    # once if we know that the axes aren't going to change (otherwise we'd do
    # this in the loop)
    data = osc.get_data()

    # Set up the plotting parameters
    plt.ion()
    plt.show()
    plt.grid(visible=True)
    plt.ylim([-1, 1])
    plt.xlim([data['time'][0], data['time'][-1]])

    (line1,) = plt.plot([])
    (line2,) = plt.plot([])

    # Configure labels for axes
    ax = plt.gca()

    # This loops continuously updates the plot with new data
    while True:
        # Get new data
        data = osc.get_data()

        # Update the plot
        line1.set_ydata(data['ch1'])  # Streamed data from Gigabit Streamer in Slot 1 (blue)
        line1.set_xdata(data['time'])
        line2.set_ydata(data['ch2'])  # Signal generated in Oscilloscope in loopback (orange)
        line2.set_xdata(data['time'])
        plt.pause(0.001)

except Exception as e:
    m.relinquish_ownership()
    raise e
finally:
    m.relinquish_ownership()
