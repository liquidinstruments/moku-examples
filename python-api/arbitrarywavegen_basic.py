#
# moku example: Arbitrary waveform generator
#
# This example demonstrates how you can generate and output arbitrary
# waveforms using Moku AWG
#
# (c) Liquid Instruments Pty. Ltd.
#
import numpy as np

from moku.instruments import ArbitraryWaveformGenerator

# Generate a signal that the Arbitrary Waveform Generator should generate on the output
t = np.linspace(0, 1, 100)  # Evaluate our waveform at 100 points

# Simple square wave (can also use scipy.signal)
sq_wave = np.array([-1.0 if x < 0.5 else 1.0 for x in t])

# More interesting waveform. Note that we have to normalize this waveform
# to the range [-1, 1]
not_sq = np.zeros(len(t))
for h in np.arange(1, 15, 2):
    not_sq += (4 / (np.pi * h)) * np.cos(2 * np.pi * h * t)

not_sq = not_sq / max(abs(not_sq))

# Connect to your Moku by its ip address ArbitraryWaveformGenerator('192.168.###.###')
# force_connect will overtake an existing connection
i = ArbitraryWaveformGenerator('192.168.###.###', force_connect=True)

try:
    # Load and configure the waveform.
    i.generate_waveform(
        channel=1, sample_rate='Auto', lut_data=list(sq_wave), frequency=10e3, amplitude=1
    )
    i.generate_waveform(
        channel=2, sample_rate='Auto', lut_data=list(not_sq), frequency=10e3, amplitude=1
    )

    # Set channel 1 to pulse mode
    # 2 dead cycles at 0Vpp
    i.pulse_modulate(channel=1, dead_cycles=2, dead_voltage=0)

    # Set Channel 2 to burst mode
    # Burst mode triggering from Input 1 at 0.1 V
    # 3 cycles of the waveform will be generated every time it is triggered
    i.burst_modulate(
        channel=2,
        trigger_source='Input1',
        trigger_mode='NCycle',
        burst_cycles=3,
        trigger_level=0.1,
    )

except Exception as e:
    i.relinquish_ownership()
    raise e

finally:
    # Close the connection to the Moku device
    # This ensures network resources and released correctly
    i.relinquish_ownership()
