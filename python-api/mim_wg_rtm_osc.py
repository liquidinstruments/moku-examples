# Multi-instrument Waveform Generator, Real Time Math and Oscilloscope example
#
# This example demonstrates the Real Time Math (RTM) instrument in
# Multi-Instrument Mode. A Waveform Generator in slot 1 drives RTM in slot 2,
# and an Oscilloscope in slot 3 shows the two RTM outputs:
#   - Output A: the signal amplified by 6 dB, with the output limited to +/-0.5 V
#   - Output B: the signal with 0.25 V added to it
#
# (c) Liquid Instruments Pty. Ltd.

import matplotlib.pyplot as plt

from moku.instruments import MultiInstrument, Oscilloscope, RealTimeMath, WaveformGenerator

# Replace with your Moku's IP address.
MOKU_IP = '192.168.###.###'

# RTM only runs in Multi-Instrument Mode. This example uses three slots, so use
# 3 for Moku:Go, Moku:Lab and Moku:Delta, or 4 for Moku:Pro.
NUM_SLOTS = 3

# Connect to the Moku (force_connect will overtake an existing connection).
with MultiInstrument(MOKU_IP, num_slots=NUM_SLOTS, force_connect=True) as m:
    wg = m.set_instrument(1, WaveformGenerator)
    rtm = m.set_instrument(2, RealTimeMath)
    osc = m.set_instrument(3, Oscilloscope)

    # Waveform Generator -> RTM -> Oscilloscope
    connections: list[MultiInstrument.Connection] = [
        {"source": "Slot1OutA", "destination": "Slot2InA"},
        {"source": "Slot2OutA", "destination": "Slot3InA"},
        {"source": "Slot2OutB", "destination": "Slot3InB"},
    ]
    m.set_connections(connections=connections)

    # A 1 Vpp, 10 kHz sine wave
    wg.generate_waveform(1, "Sine", amplitude=1, frequency=10e3)

    # Each RTM output channel applies a function to the input signals.
    # Output A: gain of 6 dB (about x2), limited to +/-0.5 V so the peaks are clipped
    rtm.set_node(channel=1, column=1, input_a=1, function="Gain", gain=6)
    rtm.set_output_limit(channel=1, enable=True, low_limit=-0.5, high_limit=0.5)

    # Output B: add a 0.25 V offset to the signal
    rtm.set_node(channel=2, column=1, input_a=1, function="Offset", offset=0.25)

    # Print the RTM configuration
    print(rtm.summary())

    # View both outputs on the oscilloscope
    osc.set_timebase(-0.5e-3, 0.5e-3)
    data = osc.get_data()

    plt.plot(data['time'], data['ch1'], label="RTM output A (gain, limited)")
    plt.plot(data['time'], data['ch2'], label="RTM output B (offset)")
    plt.grid(visible=True)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (V)")
    plt.legend()
    plt.show()
