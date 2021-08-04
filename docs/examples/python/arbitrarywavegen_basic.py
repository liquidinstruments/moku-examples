#
# moku example: Arbitrary waveform generator
#
# This example demonstrates how you can generate and output arbitrary
# waveforms using Moku AWG
#
# (c) 2021 Liquid Instruments Pty. Ltd.
#
import numpy as np
from moku.instruments import ArbitraryWaveformGenerator

# generate a signal that the Arbitrary Waveform Genenerator should generate on the output
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
# or by its serial ArbitraryWaveformGenerator(serial=123)
i = ArbitraryWaveformGenerator('192.168.###.###', force_connect=False)
i.name()
try:
    # Load and configure the waveform.
    i.generate_waveform(channel=1, sample_rate='Auto',
                        lut_data=list(sq_wave), frequency=10e3,
                        amplitude=1)
    i.generate_waveform(channel=2, sample_rate='Auto', lut_data=list(not_sq),
                        frequency=10e3, amplitude=1)

except Exception as e:
    print(f'Exception occurred: {e}')

finally:
    # Close the connection to the Moku device
    # This ensures network resources and released correctly
    i.relinquish_ownership()
