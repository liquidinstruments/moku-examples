# Simple Adder Example

This example assigns the outputs A and B as the sum and difference of the inputs A and B.

OutputA is Input A + Input B;

OutputB is Input A - Input B;

<code-group>

<code-block title='VHDL'>

<<< @/docs/api/moku-examples/mcc/Basic/Adder/Adder.vhd

</code-block>

<code-block title='Verilog'>

```verilog
module CustomWrapper (
    input wire Clk,
    input wire Reset,
    input wire [31:0] Sync,

    input wire signed [15:0] InputA,
    input wire signed [15:0] InputB,
    input wire signed [15:0] InputC,
    input wire signed [15:0] InputD,

    input wire ExtTrig,

    output wire signed [15:0] OutputA,
    output wire signed [15:0] OutputB,
    output wire signed [15:0] OutputC,
    output wire signed [15:0] OutputD,

    output wire OutputInterpA,
    output wire OutputInterpB,
    output wire OutputInterpC,
    output wire OutputInterpD,

    input wire [31:0] Control0,
    input wire [31:0] Control1,
    input wire [31:0] Control2,
    input wire [31:0] Control3,
    input wire [31:0] Control4,
    input wire [31:0] Control5,
    input wire [31:0] Control6,
    input wire [31:0] Control7,
    input wire [31:0] Control8,
    input wire [31:0] Control9,
    input wire [31:0] Control10,
    input wire [31:0] Control11,
    input wire [31:0] Control12,
    input wire [31:0] Control13,
    input wire [31:0] Control14,
    input wire [31:0] Control15
);

assign OutputA = InputA + InputB;
assign OutputB = InputA - InputB;
endmodule
```

</code-block>

</code-group>

# Voltage limiter example

This example uses the clip function from the Moku Library to limit the output signal to a set range. The upper limit of Output A is set by Control0, the lower limit of Output A is set by Control1.  The upper limit of Output B is set by Control2, the lower limit of Output B is set by Control3.  

<<< @/docs/api/moku-examples/mcc/Basic/VoltageLimiter/limiter.vhd

# DSP example

This example instantiates a DSP block using the [ScaleOffset](../support.md#scaleoffset) wrapper. The `Moku.Support.ScaleOffset` entity conveniently packages a DSP block with all the settings configured to compute the common `Z = X * Scale + Offset` operation, with the output properly clipped to prevent under/overflow.

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
