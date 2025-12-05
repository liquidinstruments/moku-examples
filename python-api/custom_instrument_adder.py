#
# moku example: Adder Custom Instrument
#
# This example demonstrates how you can configure Custom Instrument, using
# Multi-Instrument mode, to add and subtract two input signals together and
# output the result to the Oscilloscope.
#
#  (c) Liquid Instruments Pty. Ltd.
#

import matplotlib.pyplot as plt

from moku.instruments import CustomInstrument, MultiInstrument, Oscilloscope

# Connect to your Moku by its ip address using
# MultiInstrument('192.168.###.###')
# force_connect will overtake an existing connection
m = MultiInstrument('192.168.###.###', platform_id=2, force_connect=True)

try:
    # Set the instruments and upload Custom Instrument bitstreams from your device
    # to your Moku
    bitstream = "path/to/project/adder/bitstreams.tar.gz"
    mc = m.set_instrument(1, CustomInstrument, bitstream=bitstream)
    osc = m.set_instrument(2, Oscilloscope)

    # Configure the connections
    connections = [
        dict(source="Slot1OutA", destination="Slot2InA"),
        dict(source="Slot1OutB", destination="Slot2InB"),
        dict(source="Slot2OutA", destination="Slot1InA"),
        dict(source="Slot2OutB", destination="Slot1InB"),
    ]

    m.set_connections(connections=connections)

    # Configure the Oscilloscope to generate a ramp wave and square wave with
    # equal frequencies, then sync the phases
    osc.generate_waveform(1, 'Square', amplitude=1, frequency=1e3, duty=50)
    osc.generate_waveform(2, 'Ramp', amplitude=1, frequency=1e3, symmetry=50)
    osc.sync_output_phase()

    # Set the time span to cover four cycles of the waveforms
    osc.set_timebase(-2e-3, 2e-3)

    # Retrieve data
    data = osc.get_data()

    # Set up the plotting parameters
    plt.plot(data['time'], data['ch1'], label='Add')
    plt.plot(data['time'], data['ch2'], label='Subtract')

    # Configure labels for axes
    plt.xlabel("Time [Second]")
    plt.ylabel("Amplitude [Volt]")
    plt.grid(visible=True)
    plt.legend(loc=0)
    plt.show()

except Exception as e:
    m.relinquish_ownership()
    raise e
finally:
    m.relinquish_ownership()
