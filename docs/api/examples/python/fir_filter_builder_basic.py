#
# moku example: Basic FIR Filter Builder
#
# This example demonstrates how to run the FIR Filter Box and configure its
# individual filter channel coefficients.
#
# (c) 2024 Liquid Instruments Pty. Ltd.
#

from moku.instruments import FIRFilterBox

# The following two example arrays are simple rectangular FIR kernels with 50
# and 400 taps respectively. A rectangular kernel produces a sinc shaped
# transfer function with width inversely proportional to the length of the
# kernel. FIR kernels must have a normalized power of <= 1.0, so the value of
# each tap is the inverse of the total number of taps.
filter1 = [1.0 / 50.0] * 50
filter2 = [1.0 / 400.0] * 400

# Connect to your Moku by its ip address FIRFilterBox('192.168.###.###')
# or by its serial FIRFilterBox(serial=123)
i = FIRFilterBox("192.168.###.###", force_connect=True)

try:
    # Configure the Moku's frontend settings
    i.set_frontend(1, impedance="1MOhm", attenuation="0dB", coupling="DC")
    i.set_frontend(2, impedance="1MOhm", attenuation="0dB", coupling="DC")

    # Load the coefficients and sample rate for each FIR filter channel.
    # To implement 50 FIR taps
    i.set_custom_kernels(1, sample_rate="1.953MHz", coefficients=filter1)
    # To implement 400 FIR taps
    i.set_custom_kernels(2, sample_rate="1.953MHz", coefficients=filter2)

    # Both channels have unity gain and no offsets
    i.set_input_gain(1, gain=1.0)
    i.set_output_gain(1, gain=1.0)
    i.set_input_offset(1, offset=0.0)
    i.set_input_gain(2, gain=1.0)
    i.set_output_gain(2, gain=1.0)
    i.set_input_offset(2, offset=0.0)

except Exception as e:
    print(f"Exception Occurred: {e}")

finally:
    i.relinquish_ownership()