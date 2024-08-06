#
# moku example: FIR Filter Builder Plotting Example
#
# This script demonstrates how to generate an FIR filter kernel with specified
# parameters using the scipy library, and how to configure settings of the FIR
# instrument.
#
#
# (c) 2023 Liquid Instruments
#

from moku.instruments import FIRFilterBox
from scipy.fft import fft
import math
import matplotlib.pyplot as plt

# Specify Nyquist and cutoff (-3dB) frequencies
nyq_rate = 125e6 / 2**10 / 2.0
cutoff_hz = 1e3

# Calculate FIR kernel using 1000 taps and a Chebyshev window with -60dB
# stop-band attenuation
taps = [cutoff_hz / nyq_rate] * 1000

# Connect to your Moku by its ip address FIRFilterBox('192.168.###.###')
# or by its serial FIRFilterBox(serial=123)
i = FIRFilterBox("192.168.###.###", force_connect=True)

try:
    # Configure the Moku frontend settings
    i.set_frontend(1, impedance="50Ohm", attenuation="0dB", coupling="DC")
    i.set_frontend(2, impedance="50Ohm", attenuation="0dB", coupling="DC")

    # Both filter channels are configured with the same FIR kernel. A
    # decimation factor of 10 is used to achieve the desired Nyquist rate and
    # FIR kernel length of 1000.
    i.set_custom_kernel_coefficients(1, sample_rate="2.441MHz", coefficients=taps)
    i.set_custom_kernel_coefficients(2, sample_rate="2.441MHz", coefficients=taps)

    # Channel 1 has unity input/output gain and acts solely on ADC1.
    # Channel 2 has an input gain of 0.5, output gain of 2.0, input offset of
    # -0.1V and acts on signal 0.5 * ADC1 + 0.5 * ADC2.
    i.set_input_gain(1, gain=1.0)
    i.set_output_gain(1, gain=1.0)
    i.set_input_gain(2, gain=0.5)
    i.set_output_gain(2, gain=1.0)
    i.set_input_offset(2, offset=-0.1)
    i.set_control_matrix(1, 1.0, 0.0)
    i.set_control_matrix(2, 0.5, 0.5)

    # Set which signals to view on each monitor channel, and the timebase on
    # which to view them.
    i.set_timebase(-5e-3, 5e-3)
    i.set_monitor(1, "Input1")
    i.set_monitor(2, "Output1")

    # Calculate and plot the quantized FIR kernel and transfer function for
    # reference.
    taps_quantized = [round(taps[x] * 2.0**24 - 1) / (2**24 - 1) for x in range(0, len(taps))]
    fft_taps = fft(taps_quantized)
    fft_mag = [abs(fft_taps[x]) for x in range(0, len(fft_taps[0:499]))]
    epsilon = 1e-14
    fft_db = [20 * math.log10(max(fft_mag[x], epsilon)) for x in range(0, len(fft_mag))]

    plt.subplot(221)
    plt.plot(taps)
    plt.title("Filter Kernel")
    plt.ylabel("Normalized Value")
    plt.grid(True, which="major")
    plt.xlabel("Kernel Tap Number")
    plt.subplot(222)
    plt.semilogx(fft_db)
    plt.title("Filter Transfer Function")
    plt.ylabel("Magnitude (dB)")
    plt.xlabel("Frequency (Hz)")
    plt.grid(True, which="major")

    # Set up the live FIR Filter Box monitor signal plot
    plt.subplot(212)
    plt.title("Monitor Signals")
    plt.suptitle("FIR Filter Box", fontsize=16)
    plt.grid(True, which="both", axis="both")
    data = i.get_data()  # Get data to determine the signal timebase
    dt = data["time"]
    plt.xlim([dt[0], dt[-1]])
    plt.ylim([-1.0, 1.0])  # View up to +-1V
    plt.tight_layout()

    line1, = plt.plot([], label='Input 1')
    line2, = plt.plot([], label='Output 1')
    plt.legend(handles=[line1, line2], loc=1)

    # Continually update the monitor signal data being displayed
    while True:
        data = i.get_data()
        line1.set_ydata(data["ch1"])
        line2.set_ydata(data["ch2"])
        line1.set_xdata(data["time"])
        line2.set_xdata(data["time"])
        plt.pause(0.001)

except Exception as e:
    print(f"Exception Occurred: {e}")
finally:
    i.relinquish_ownership()
