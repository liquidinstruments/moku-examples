# Status Registers

The [CustomInstrument entity](./wrapper.html#custominstrument-architecture) provides 16 status registers which can be used to monitor the behavior of the custom design at runtime. The registers are defined as an array of 32-bit std_logic_vectors. These are labelled as **Status** and are defined only in the CustomInstrument entity.

## Type Casting

These status registers can be assigned to different internal signals or control registers to playback these values in real-time. With VHDL code, assigning to the status registers will often require casting to std_logic_vector. When using Verilog (SystemVerilog), the casting and resizing is often implicit during the assignment and is done automatically.

<code-group>

<code-block title="VHDL">

```vhdl
-- Import libraries that contain the types we need
library IEEE;
use IEEE.Std_Logic_1164.all;  -- for std_logic(_vector) and resize()
use IEEE.Numeric_Std.all;  -- for signed and unsigned etc.

architecture Behavioural of CustomInstrument is
    -- define signals here
    signal A : signed(31 downto 0);
    signal B : unsigned(11 downto 0);
begin

    Status(0)(15 downto 0) <= Control(0)(15 downto 0);  ---- take 16 LSBs and read back what is set in Control Register
    A <= signed(Control(0)0(15 downto 0)) + signed(Control(1)(15 downto 0));    ---- add two numbers given through Control registers
    Status(1)(31 downto 0) <= std_logic_vector(A);  -----typecasting to std_logic_vector to read the sum as status register
    Status(2) <= std_logic_vector(resize(unsigned(B), 32)); ----resizing and type casting to read as status register.

end architecture;
```

</code-block>

<code-block title="Verilog">

```verilog
module CustomInstrument (
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

    input wire [31:0] Control [0:15];
    output wire [31:0] Status [0:15];
);
reg signed [15:0] A;
reg unsigned [63:0] B;

assign Status[0][12:0] = Control[0][12:0];  // bit slicing and automatic casting as signed
assign A = Control[1][15:0] + Control[2][15:0];
assign Status[1][15:0] = A;    // can assigns the same control
                               // register to another variable
assign Status[2] = B;          // Automatic resized and casting when assigning

endmodule
```

</code-block>

</code-group>
