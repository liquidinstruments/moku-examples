# Basic Examples

## Simple Adder

This example assigns the outputs A and B as the sum and difference of the inputs A and B.

Output A is Input A + Input B;

Output B is Input A - Input B;

<code-group>

<code-block title='VHDL'>

```vhdl
-- A very simple example, simply add two inputs and route to an output.
-- This is purely combinatorial
architecture Behavioural of CustomWrapper is
begin
    OutputA <= InputA + InputB;
    OutputB <= InputA - InputB;
end architecture;
```

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

<action-button text="Adder | GitHub" link="https://github.com/liquidinstruments/moku-examples/tree/main/mcc/Basic/Adder" target="_blank"/>

## Voltage limiter

This example uses the clip function from the Moku Library to limit the output signal to a set range. The upper limit of Output A is set by Control0, the lower limit of Output A is set by Control1.  The upper limit of Output B is set by Control2, the lower limit of Output B is set by Control3.  

```vhdl
library IEEE;
use IEEE.Numeric_Std.all;
library Moku;
use Moku.Support.clip_val;
use Moku.Support.sum_no_overflow;

architecture Behavioural of CustomWrapper is
    signal ch1_lower : signed(15 downto 0);
    signal ch1_upper : signed(15 downto 0);
    signal ch2_lower : signed(15 downto 0);
    signal ch2_upper : signed(15 downto 0);

begin
    ch1_lower <= signed(Control0(15 downto 0));
    ch1_upper <= signed(Control1(15 downto 0));
    ch2_lower <= signed(Control2(15 downto 0));
    ch2_upper <= signed(Control3(15 downto 0));

    -- Use library function to "clip" the value to within the upper and lower bounds
    OutputA <= clip_val(InputA, to_integer(ch1_lower), to_integer(ch1_upper));
    OutputB <= clip_val(InputB, to_integer(ch2_lower), to_integer(ch2_upper));
end architecture;
```

<action-button text="Voltage Limiter | GitHub" link="https://github.com/liquidinstruments/moku-examples/tree/main/mcc/Basic/VoltageLimiter" target="_blank"/>

## DSP

This example instantiates a DSP block using the [ScaleOffset](../support.md#scaleoffset) wrapper. The `Moku.Support.ScaleOffset` entity conveniently packages a DSP block with all the settings configured to compute the common `Z = X * Scale + Offset` operation, with the output properly clipped to prevent under/overflow.

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

```vhdl
library IEEE;
use IEEE.Numeric_Std.all;

library Moku;
use Moku.Support.ScaleOffset;

-- Instantiate a DSP block using the ScaleOffset wrapper
architecture Behavioural of CustomWrapper is
begin
    -- Z = X * Scale + Offset
    -- Offset is units of bits, scale by default runs from -1 to 1 across whatever signal width is given
    -- Clips Z to min/max (prevents over/underflow)
    -- Includes rounding
    -- One Clock Cycle Delay
    DSP: ScaleOffset
        port map (
            Clk => Clk,
            Reset => Reset,
            X => InputA,
            Scale => signed(Control0(15 downto 0)),
            Offset => signed(Control1(15 downto 0)),
            Z => OutputA,
            Valid => '1',
            OutValid => open
        );

    -- If you want to change the range of the scale (e.g. multiply by more than 1), then set the
    -- NORMAL_SHIFT generic. This increases the range of Scale by 2^N, so NORMAL_SHIFT=4 means that
    -- the 16 bit scale here now covers the range -16 to 16.
    DSP_RANGE: ScaleOffset
        generic map (
            NORMAL_SHIFT => 4
        )
        port map (
            Clk => Clk,
            Reset => Reset,
            X => InputB,
            Scale => signed(Control2(15 downto 0)),
            Offset => signed(Control3(15 downto 0)),
            Z => OutputB,
            Valid => '1',
            OutValid => open
        );
end architecture;
```

<action-button text="DSP | GitHub" link="https://github.com/liquidinstruments/moku-examples/tree/main/mcc/Basic/DSP" target="_blank"/>
