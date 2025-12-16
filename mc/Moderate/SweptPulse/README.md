# Moku Compile Example - Swept Frequency Pulse

The example code in this directory is designed to highlight the following capabilities:

- [Python API](https://apis.liquidinstruments.com/api/) configuration with multiple Moku devices
- [Custom Instrument](https://apis.liquidinstruments.com/mc/) design and incorporation via Multi-instrument Mode
- Python ability to load and process data logger files


## Overview
A swept frequency pulse is often referred to as a chirped pulse, and is commonly used in radar applications.  A chirped pulse is one where the transmitted frequency continuously changes (sweeps) throughout the duration of the pulse.  While the [Waveform Generator](https://liquidinstruments.com/products/integrated-instruments/waveform-generator/)  on the Moku allows for the creation of swept waveforms, it cannot by default create a chirped (swept waveform) pulse.  However, with Moku Compile we can quite easily add this functionality into the Moku.  We will use the Moku in Multi-instrument Mode along with the Python API and Custom Instrument to demonstrate the added flexibility.  

## Included Files
The following files are included to aid with reproducing the results highlighted below.  Due to minor variations in required configuration, the appropriate files for each Moku instrument are stored in a separate folder.

### Moku:Go
Using the Moku:Go, we can use a single device to generate the swept pulse, but an additional device would be required to display the output.  The pulse timing wave will be output to channel 1 and fed back into Input A of Slot 1 to start the sweep.  The swept pulse will be output to channel 2.  The Moku:Go also has the ability to receive a reset signal and output the pulse mask to a Digital Input / Output (DIO) pin.

- **mim\_mgo\_wg\_mc.py** will set up the Multi-instrument mode environment on the Moku:Go.  **You will need to edit lines 21 and 26 for the actual IP of your device and filename of the bitstream created with the following VHDL files**
- **Top.vhd** designs the entity to create a variable frequency and duty cycle pulse mask that will work in conjunction with the Waveform Generator to output a swept frequency pulse
- **PulseMask.vhd** describes the behavior of the PulseMask entity

![image](images/MGoMiMConfiguration.png)

### Moku:Lab
Using the Moku:Lab, we can use a single device to generate the swept pulse, but an additional device would be required to display the output.  The pulse timing wave will be output to channel 1 and fed back into Input A of Slot 1 to start the sweep.  The swept pulse will be output to channel 2.   

- **mim\_mlab\_wg\_mc.py** will set up the Multi-instrument mode environment on the Moku:Lab.  **You will need to edit lines 21 and 26 for the actual IP of your device and filename of the bitstream created with the following VHDL files**
- **Top.vhd** designs the entity to create a variable frequency and duty cycle pulse mask that will work in conjunction with the Waveform Generator to output a swept frequency pulse
- **PulseMask.vhd** describes the behavior of the PulseMask entity

![image](images/MLabMiMConfiguration.png)

### Moku:Pro
Using the Moku:Pro, we can use the additional available slots to not only generate the swept pulse, but also display and record the signal.  As seen in the image below, the configuration is far more complex, viewing/recording can all be done with a single device.  The pulse could be output to one of the four Moku output channels if desired with a simple modification to the included API script. 

- **mim\_mlab\_wg\_mc\_la\_osc.py** will set up the Multi-instrument mode environment on the Moku:Pro.  **You will need to edit lines 30 and 34 for the actual IP of your device and filename of the bitstream created with the following VHDL files**
- **Top.vhd** designs the entity to create a variable frequency and duty cycle pulse mask that will work in conjunction with the Waveform Generator to output a swept frequency pulse
- **PulseMask.vhd** describes the behavior of the PulseMask entity

![image2](images/MProMiMConfiguration.png)

## Example Pulse
The following pulse is representative of the type of signal that would be used in a basic swept pulse system / RADAR.  The included API scripts will allow the user to specify the pulse parameters below

### Example Pulse parameters
- Starting Frequency - 1kHz
- Ending Frequency - 20kHz
- Pulse Repetition Frequency - 100Hz
- Duty Cycle - 20%

![image](images/Results.png)

