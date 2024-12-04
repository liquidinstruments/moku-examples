# Moku Neural Network Example - Pulsed RADAR

The example code in this directory is designed to highlight the following capabilities:

- Application of [autoencoder](https://apis.liquidinstruments.com/mnn/examples/Autoencoder.html) Neural Network Capability
- Python API configuration of Neural Network and Multi Instrument Mode
- Python ability to load and process data logger files
- Python API configuration with multiple Moku devices

## Overview
In this example we will use a Moku:Go running the waveform generator to first generate a pulsed waveform.  This pulsed waveform is designed to represent the type of data commonly used in RADAR applications.  The pulsed waveform will then be received by a Moku:Pro running in multi instrument mode.  We will use the control matrix within the PID controller to add noise to the waveform, the neural network instrument configured according to the [autoencoder](https://apis.liquidinstruments.com/mnn/examples/Autoencoder.html) example to de-noise the signal.  Finally we will use the data logger to store data samples in .csv format for follow on processing and analysis.

## Example Pulse
The following pulse is representative of the type of signal that would be used in a basic pulsed RADAR.  There are many more complex techniques often employed, but for purposes of this example we will stick with the basic pulsed waveform

### Pulse parameters
- Carrier Frequency - 2kHz
- Pulse Width - 5ms
- Pulse Repetition Frequency - 10Hz

