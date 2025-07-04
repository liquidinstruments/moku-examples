# Moku Example: Basic FIR Filter Builder Configuration ##

# This example demonstrates how to configure the FIR filter builder
# and how to set the frequency, shape, window_functions, and other
# parameters to the filter. 

# (c) 2025 Liquid Instruments Pty. Ltd. ##


from moku.instruments import FIRFilterBox

# The first step in configuring the FIR filter is to connect to your Moku using the IP address. 
# Then, configure the front end and outputs of the FIR filter.

i=FIRFilterBox('192.168.2.200', force_connect=True)

try:
    # For Moku:Pro and Moku:Lab, the input impedance can be set to 50 Ohm
    # or 1M Ohm, but the Moku:Go can only be set to 1MOhm. 

    i.set_frontend(channel=1, impedance='1MOhm', attenuation='0dB', coupling='DC')
    i.enable_output(channel=1, signal=True, output=True)
    i.set_control_matrix(channel=1,input_gain1=1,input_gain2=0)
    i.set_input_gain(channel=1,gain=0)
    i.set_input_offset(channel=1,offset=0)
    i.set_output_gain(channel=1,gain=0)
    i.set_output_offset(channel=1,offset=0)

    print("Successfully configured FIR input and output.")

    # The cutoff frequencies of the filter are defined as a fraction of the sampling rate. For example, 
    # if the sample rate is 39.06MHz and the corner frequency is set to 0.1, then the cutoff frequency is 3.906MHz.


    # Sample Rate: 39.06 MHz
    # SR * corner_val = corner_freq

    # 39.06 MHz * 0.1 = 3.906 MHz
    # 39.06 MHz * 0.4 = 15.624 MHz

    #Lowpass filter corner frequency
    f_low=0.1 # 3.906 MHz

    #Highpass filter corner frequency:
    f_high=0.4 # 15.624 MHz

    #Bandpass filter low and high corner frequency
    bp_low=0.1 # 3.906 MHz
    bp_high=0.4 # 15.624 MHz

    #Bandstop filter low and high corner frequency
    bs_low=0.1 # 3.906 MHz
    bs_high=0.4 # 15.624 MHz

    # On Moku:Pro, there are four available channels, but on Moku:Lab and Go, there are only two available channels. The max sample rate
    # for the Moku:Pro is 39.06 MHz, for the Moku:Lab it is 15.63 MHz, and for Moku:Go it is 15.63 MHz. Because of this, the corner
    # frequency calculations will look different depending on the hardware being used.

    i.set_by_frequency(channel=1, sample_rate='39.06MHz', coefficient_count=201,
                        shape='Lowpass', low_corner=f_low, window='Blackman')
    i.set_by_frequency(channel=2, sample_rate='39.06MHz', coefficient_count=201,
                        shape='Highpass', high_corner=f_high, window='Bartlett')
    i.set_by_frequency(channel=3, sample_rate='39.06MHz', coefficient_count=201,
                        shape='Bandpass', low_corner=bp_low, high_corner=bp_high, window='Hann')
    i.set_by_frequency(channel=4, sample_rate='39.06MHz', coefficient_count=201,
                        shape='Bandstop', low_corner=bs_low, high_corner=bs_high, window='Hamming')

except Exception as e:
    print(f'Exception occurred: {e}')
else:
    print('Filters configured successfully.')
finally:
    i.relinquish_ownership()