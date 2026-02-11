# Basic Examples

## Simple Adder

This example assigns the outputs A and B as the sum and difference of the inputs
A and B.

Output A is Input A + Input B;

Output B is Input A - Input B;

<code-group>

<code-block title="VHDL">

<<< @/docs/api/moku-examples/mc/Basic/Adder/VHDL/Adder.vhd

</code-block>

<code-block title="Verilog">

<<< @/docs/api/moku-examples/mc/Basic/Adder/Verilog/Adder.v

</code-block>

</code-group>

<open-in-moku-compile
    name="Adder"
    githubPath="Basic/Adder"
/>

## Voltage limiter

This example uses the clip function from the Moku Library to limit the output
signal to a set range. The upper limit of Output A is set by Control0, the lower
limit of Output A is set by Control1.  The upper limit of Output B is set by
Control2, the lower limit of Output B is set by Control3.

<code-group>

<code-block title="VHDL">

<<< @/docs/api/moku-examples/mc/Basic/VoltageLimiter/limiter.vhd

</code-block>

</code-group>

<open-in-moku-compile
    name="Voltage limiter"
    githubPath="Basic/VoltageLimiter"
/>

## DSP

This example instantiates a DSP block using the
[ScaleOffset](../support.md#scaleoffset) wrapper. The `Moku.Support.ScaleOffset`
entity conveniently packages a DSP block with all the settings configured to
compute the common `Z = X * Scale + Offset` operation, with the output properly
clipped to prevent under/overflow.

### Getting Started

#### Signals and Settings

| Port     | Use                       |
| -------- | ------------------------- |
| Control0 | Scale A                   |
| Control1 | Offset A                  |
| Control2 | Scale B                   |
| Control3 | Offset B                  |
| Output A | Scaled and Offset Input A |
| Output B | Scaled and Offset Input B |

<code-group>

<code-block title="VHDL">

<<< @/docs/api/moku-examples/mc/Basic/DSP/DSP.vhd

</code-block>

</code-group>

<open-in-moku-compile
    name="DSP"
    githubPath="Basic/DSP"
/>
