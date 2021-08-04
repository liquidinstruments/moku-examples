#
# moku example: Plotting Frequency Response Analyzer
#
# This example demonstrates how you can generate output sweeps using the
# Frequency Response Analyzer instrument, and view transfer function data
# in real-time.
#
# (c) 2021 Liquid Instruments Pty. Ltd.
#
import matplotlib.pyplot as plt
from moku.instruments import FrequencyResponseAnalyzer

# Connect to your Moku by its ip address using FrequencyResponseAnalyzer('192.168.###.###')
# or by its serial number using FrequencyResponseAnalyzer(serial=123)
i = FrequencyResponseAnalyzer('192.168.###.###', force_connect=False)

# Define output sweep parameters here for readability
f_start = 20e6  # Hz
f_end = 100  # Hz
sweep_length = 512
log_scale = True
amp_ch1 = 0.5  # Vpp
amp_ch2 = 0.5  # Vpp
averaging_time = 1e-6  # sec
settling_time = 1e-6  # sec
averaging_cycles = 1
settling_cycles = 1

try:
    # Set the output sweep amplitudes
    i.set_output(1, amp_ch1)
    i.set_output(2, amp_ch2)

    # Set the sweep configuration
    i.set_sweep(start_frequency=f_start, stop_frequency=f_end,
                num_points=sweep_length, averaging_time=averaging_time,
                settling_time=settling_time, averaging_cycles=averaging_cycles,
                settling_cycles=settling_cycles)

    # Start the output sweep in loop mode
    i.start_sweep()

    # Set up the amplitude plot
    plt.subplot(211)
    if log_scale:
        # Plot log x-axis if frequency sweep scale is logarithmic
        line1, = plt.semilogx([])
        line2, = plt.semilogx([])
    else:
        line1, = plt.plot([])
        line2, = plt.plot([])
    ax_1 = plt.gca()
    ax_1.set_xlabel('Frequency (Hz)')
    ax_1.set_ylabel('Magnitude (dB)')

    # Set up the phase plot
    plt.subplot(212)
    if log_scale:
        line3, = plt.semilogx([])
        line4, = plt.semilogx([])
    else:
        line3, = plt.plot([])
        line4, = plt.plot([])
    ax_2 = plt.gca()
    ax_2.set_xlabel('Frequency (Hz)')
    ax_2.set_ylabel('Phase (Cycles)')

    plt.ion()
    plt.show()
    plt.grid(b=True)

    # Retrieves and plot new data
    while True:
        frame = i.get_data()
        ch1Data = frame['ch1']
        ch2Data = frame['ch2']

        # Set the frame data for each channel plot
        plt.subplot(211)

        ch1MagnitudeData = ch1Data['magnitude']
        ch2MagnitudeData = ch2Data['magnitude']
        line1.set_ydata(ch1MagnitudeData)
        line2.set_ydata(ch2MagnitudeData)
        line1.set_xdata(ch1Data['frequency'])
        line2.set_xdata(ch2Data['frequency'])

        # Phase
        plt.subplot(212)
        ch1PhaseData = ch1Data['phase']
        ch2PhaseData = ch2Data['phase']
        line3.set_ydata(ch1PhaseData)
        line4.set_ydata(ch2PhaseData)
        line3.set_xdata(ch1Data['frequency'])
        line4.set_xdata(ch2Data['frequency'])

        # Ensure the frequency axis is a tight fit
        ax_1.set_xlim(min(ch1Data['frequency']), max(ch2Data['frequency']))
        ax_2.set_xlim(min(ch1Data['frequency']), max(ch2Data['frequency']))
        ax_1.relim()
        ax_1.autoscale_view()
        ax_2.relim()
        ax_2.autoscale_view()

        # Redraw the lines
        plt.draw()

        plt.pause(0.001)

except Exception as e:
    print(f'Exception occurred: {e}')
finally:
    # Close the connection to the Moku device
    # This ensures network resources and released correctly
    i.relinquish_ownership()
