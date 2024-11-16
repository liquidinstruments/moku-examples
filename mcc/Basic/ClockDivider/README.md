# Clock Divider

This example is designed for the Moku:Go with no changes required.  It could be adapted for the Moku:Pro or Moku:Lab, but would need to output the pulse to an available DAC instead of the DIO port.

## Multi-instrument Mode Configuration

The following configuration will allow you to test the functionality of your MCC design.  The logic analyzer is not required for proper functionality of the MCC design.

### Instrument slots and port mapping
![image](images/MiMConfiguration.png)

### DIO Configuration
![image](images/DIOConfiguration.png)

### Signals and Settings
| Port | Slot | Use |
| --- | --- | --- |
| Input A  | Slot 1 (MCC) | Reset on DIO pin 0 |
| Output A | Slot 1 (MCC) | Divided Clock Pulse on DIO pin 8 |
| Input A  | Slot 2 (Logic Analyzer) | DIO |
