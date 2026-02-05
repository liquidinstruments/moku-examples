# Control Registers

The [wrapper entity](./wrapper.html#custominstrument-architecture) provides 16 Control registers which can be used to control
the behavior of the custom design at runtime. For the CustomInstrument entity, the registers are defined as an array of 32-bit std_logic_vectors. For the CustomWrapper entity, the registers are labelled **Control0** through to **Control15** and are all 32 bit std_logic_vectors.

## Type Casting

These Controls can be assigned to various signals in a custom design. With VHDL code, assigning using Controls will often require casting to another type or resizing or both. When using Verilog (SystemVerilog), the casting and resizing is often implicit during the assignment and is done automatically.

<code-group>

<code-block title="VHDL - CustomInstrument">

```vhdl
-- Import libraries that contain the types we need
library IEEE;
use IEEE.Std_Logic_1164.all;  -- for std_logic(_vector) and resize()
use IEEE.Numeric_Std.all;  -- for signed and unsigned etc.

architecture Behavioural of CustomInstrument is
    -- define signals here
    signal A : signed(12 downto 0);
    signal B : std_logic;
    signal C : unsigned(63 downto 0);
begin

    A <= signed(Control(0)(12 downto 0));  -- take 13 LSBs and cast to signed
    B <= Control(0)(15);  -- Controls can be shared
    -- resize Control1 to 64 bits, MSBs padded with '0'
    C <= resize(unsigned(Control(1)), C'length);

end architecture;
```

</code-block>

<code-block title="Verilog - CustomInstrument">

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

    input wire [31:0] control [0:15],
    output wire [31:0] status [0:15]
);
reg signed [12:0] A;
reg B;
reg unsigned [63:0] C;

assign A = control[0][12:0];  // bit slicing and automatic casting as signed
assign B = control[0][15];    // can assigns the same control
                            // register to another variable
assign C = control[1];        // Automatic resized and casting when assigning

endmodule
```

</code-block>

<!-- </code-group>


<code-group> -->

<code-block title="VHDL - CustomWrapper">

```vhdl
-- Import libraries that contain the types we need
library IEEE;
use IEEE.Std_Logic_1164.all;  -- for std_logic(_vector) and resize()
use IEEE.Numeric_Std.all;  -- for signed and unsigned etc.

architecture Behavioural of CustomWrapper is
    -- define signals here
    signal A : signed(12 downto 0);
    signal B : std_logic;
    signal C : unsigned(63 downto 0);
begin

    A <= signed(Control0(12 downto 0));  -- take 13 LSBs and cast to signed
    B <= Control0(15);  -- Controls can be shared
    -- resize Control1 to 64 bits, MSBs padded with '0'
    C <= resize(unsigned(Control1), C'length);

end architecture;
```

</code-block>

<code-block title="Verilog - CustomWrapper">

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
reg signed [12:0] A;
reg B;
reg unsigned [63:0] C;

assign A = Control0[12:0];  // bit slicing and automatic casting as signed
assign B = Control0[15];    // can assigns the same control
                            // register to another variable
assign C = Control1;        // Automatic resized and casting when assigning

endmodule
```

</code-block>

</code-group>
