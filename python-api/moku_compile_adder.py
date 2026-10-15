#
# moku example: Adder Moku Compile instrument
#
# This example demonstrates how you can configure your Moku to run a Moku Compile
# instrument in Multi-Instrument Mode, which adds and subtracts two input signals
# and outputs the results to the Oscilloscope.
#
#  (c) Liquid Instruments Pty. Ltd.
#

import matplotlib.pyplot as plt

from moku.instruments import MokuCompile, MultiInstrument, Oscilloscope

# Replace with your Moku's IP address.
DEVICE_IP = '192.168.###.###'

# Connect to the Moku (force_connect will overtake an existing connection).
with MultiInstrument(DEVICE_IP, num_slots=2, force_connect=True) as m:
    # Set the instruments and upload Moku Compile bitstreams from your device
    # to your Moku
    bitstream = "path/to/project/adder/bitstreams.tar.gz"
    mc = m.set_instrument(1, MokuCompile, bitstream=bitstream)
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
