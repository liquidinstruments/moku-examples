# DSP example

This example instantiates a DSP block using the [ScaleOffset](https://apis.liquidinstruments.com/mcc/support.html#scaleoffset) wrapper. The `Moku.Support.ScaleOffset` entity conveniently packages a DSP block with all the settings configured to compute the common `Z = X * Scale + Offset` operation, with the output properly clipped to prevent under/overflow.

## Getting Started

### Signals and Settings
| Port | Use |
| --- | --- |
| Control0  |	Scale A |
| Control1  |	Offset A |
| Control2  |	Scale B |
| Control3  |	Offset B |
| Output A | 	Scaled and Offset Input A |
| Output B | 	Scaled and Offset Input B |
