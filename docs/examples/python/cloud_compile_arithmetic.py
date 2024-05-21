#
# moku example: Arithmetic Cloud Compile
#
#  This example demonstrates how you can configure Cloud Compile to chose to
# add, subtract or multiply two input signals using the control registers
# and output the result to the Oscilloscope.
#
#  (c) 2024 Liquid Instruments Pty. Ltd.
#

import matplotlib.pyplot as plt

from moku.instruments import MultiInstrument, CloudCompile, Oscilloscope

# Connect to your Moku by its ip address using
# MultiInstrument('192.168.###.###')
# or by its serial number using MultiInstrument(serial=123)
m = MultiInstrument('192.168.###.###', platform_id=2)

try:
    # Set the instruments and upload Cloud Compile bitstreams from your device
    # to your Moku
    bitstream = "path/to/project/arithmetic/bitstreams.tar.gz"
    mcc = m.set_instrument(1, CloudCompile, bitstream=bitstream)
    osc = m.set_instrument(2, Oscilloscope)

    # Configure the connections
    connections = [dict(source="Slot1OutA", destination="Slot2InA"),
                   dict(source="Slot2OutB", destination="Slot2InB"),
                   dict(source="Slot2OutA", destination="Slot1InA"),
                   dict(source="Slot2OutB", destination="Slot1InB")]

    m.set_connections(connections=connections)

    # Configure the Oscilloscope to generate a ramp wave and square wave with
    # equal frequencies, then sync the phases
    osc.generate_waveform(1, 'Square', amplitude=50e-3,
                          frequency=1e3, duty=50)
    osc.generate_waveform(2, 'Ramp', amplitude=50e-3,
                          frequency=1e3, symmetry=50)
    osc.sync_output_phase()

    # Set the time span to cover four cycles of the waveforms
    osc.set_timebase(-2e-3, 2e-3)
    osc.set_trigger(type="Edge", edge="Rising", level=0,
                    mode="Normal", source="ChannelB")

    # Set up the plotting figure
    fig, axs = plt.subplots(3, sharex=True)

    # Set Control Register 1 to choose to add (0b00), subtract (0b01) or
    # multiply (0b10) the input signals
    mcc.set_control(1, 0b00)
    # Retrieve the data
    data = osc.get_data(wait_reacquire=True)
    # Plot the result and configure labels for the axes
    axs[0].plot(data['time'], data['ch1'], color='r')
    axs[0].grid(visible=True)
    axs[0].set_title('Add - 0b00')

    # Repeat these steps for each option
    mcc.set_control(1, 0b01)
    data = osc.get_data(wait_reacquire=True)
    axs[1].plot(data['time'], data['ch1'], color='b')
    axs[1].grid(visible=True)
    axs[1].set_ylabel("Amplitude [Volt]")
    axs[1].set_title('Subtract - 0b01')

    mcc.set_control(1, 0b10)
    data = osc.get_data(wait_reacquire=True)
    axs[2].plot(data['time'], data['ch1'], color='y')
    axs[2].grid(visible=True)
    axs[2].set_title('Multiply - 0b10')

    # Configure labels for axes
    plt.xlabel("Time [Second]")
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f'Exception occurred: {e}')
finally:
    m.relinquish_ownership()
