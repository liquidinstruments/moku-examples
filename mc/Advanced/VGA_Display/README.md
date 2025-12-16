# VGA Waveform Display

This example implements a VGA driver, designed for Moku:Go. The two analog outputs are used for two colour channels (not full RGB) while the digital pins are configured to provide the sync signals.

The waveform to be displayed is captured from InputA on a rising edge with a configurable directly-downsampled decimation factor, making this effectively a very simple Oscilloscope.

## Pinout and Registers

### Pinout

| Pin | Use |
| --- | --- |
| Input A | Waveform input |
| Input B | Not used |
| Output A | Red channel output |
| Output B | Blue channel output |
| Output C | H_Sync (bit 0), V_Sync (bit 1) |

Route Output C to the Digital I/O and ensure that Pins 1 and 2 are configured as outputs.

### Registers

| Register | Use |
| --- | --- |
| Control2 | Direct downsample decimation factor |

