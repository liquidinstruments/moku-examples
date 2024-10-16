# VHDL code for Moving Average and Median filters for both Moku:Go and Moku:Pro

The examples implement a moving averager and a median filter as explained fully in the application note
https://www.liquidinstruments.com/blog/2022/12/15/reducing-noise-and-transients-with-custom-real-time-digital-filtering/


# Arithmetic Unit example

This arithmetic unit example can perform addition, subtraction, or multiplication of two input signals depending on a control register setting. A 2-bit control register to switch between different operations:
- 00 represents Input A + Input B;
- 01 represents Input A – Input B;
- 10 represents Input A * Input B;
- 11 will just pass Input A to Output A. 


# Voltage limiter example

This example uses the clip function from the Moku Library to limit the output signal to a set range. The upper limit of Output A is set by Control0, the lower limit of Output A is set by Control1.  The upper limit of Output B is set by Control2, the lower limit of Output B is set by Control3.  

# Input Divider

This is the implimentation of OutputA = Control0 * (InputA / InputB).

Control register 0 "Control0" must be set to 1 so the OutputA product is not 0. Control0 is a scalar of the Divider output, e.g. if Control0 = 2, the signal is scaled to 2x the original output. The output signal may have to be scaled significantly (Control0 > 100) to get a legible signal. For example in the screenshot below, Control0 was set to 6000 to get an output in the mV range.

The quotient signal can be routed to the Moku Oscilloscope in the contiguous Slot for quick viewing of the signal and to check the signal is as expected;

![CC Divider Setup](./setup.png)

![CC Divider Screenshot](./screenshot.png)

This Divider code was written with [Mathworks's HDL Coder](https://www.mathworks.com/products/hdl-coder.html).

