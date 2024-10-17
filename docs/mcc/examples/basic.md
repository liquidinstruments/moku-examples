# Simple Adder Example

This example assigns the outputs A and B as the sum and difference of the inputs A and B.

OutputA is Input A + Input B;

OutputB is Input A - Input B;

<<< @/docs/api/moku-examples/mcc/Basic/Adder/Adder.vhd


# Voltage limiter example

This example uses the clip function from the Moku Library to limit the output signal to a set range. The upper limit of Output A is set by Control0, the lower limit of Output A is set by Control1.  The upper limit of Output B is set by Control2, the lower limit of Output B is set by Control3.  

<<< @/docs/api/moku-examples/mcc/Basic/VoltageLimiter/limiter.vhd

# DSP example

This example instantiates a DSP block using the [ScaleOffset](/mcc/support#scaleoffset) wrapper. The `Moku.Support.ScaleOffset` entity conveniently packages a DSP block with all the settings configured to compute the common `Z = X * Scale + Offset` operation, with the output properly clipped to prevent under/overflow.


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


<<< @/docs/api/moku-examples/mcc/Basic/DSP/DSP.vhd