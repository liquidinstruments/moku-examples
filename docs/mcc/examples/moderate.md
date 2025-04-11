# Moderate Examples
## Moving Average and Median filters

The examples implement a moving averager and a median filter as explained fully in the application note
https://www.liquidinstruments.com/blog/2022/12/15/reducing-noise-and-transients-with-custom-real-time-digital-filtering/

<action-button text="Link to GitHub" link="https://github.com/liquidinstruments/moku-examples/tree/main/mcc/Moderate/AverageAndMedian" target="_blank"/>

## Arithmetic Unit example

This arithmetic unit example can perform addition, subtraction, or multiplication of two input signals depending on a control register setting. A 2-bit control register to switch between different operations:
- 00 represents Input A + Input B;
- 01 represents Input A – Input B;
- 10 represents Input A * Input B;
- 11 will just pass Input A to Output A. 

<action-button text="Link to GitHub" link="https://github.com/liquidinstruments/moku-examples/tree/main/mcc/Moderate/ArithmeticUnit" target="_blank"/>

## Swept Frequency Pulse

A swept frequency pulse is often referred to as a chirped pulse, and is commonly used in radar applications.  A chirped pulse is one where the transmitted frequency continuously changes (sweeps) throughout the duration of the pulse.  However, with MCC we can quite easily add this functionality into the Moku Waveform Generator. 
The following pulse is representative of the type of signal that would be used in a basic swept pulse system / RADAR.

![Swept Frequency Result](./swept_freq_result.png)

<action-button text="Link to GitHub" link="https://github.com/liquidinstruments/moku-examples/tree/main/mcc/Moderate/SweptPulse" target="_blank"/>

