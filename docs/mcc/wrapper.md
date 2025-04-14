# Custom Wrapper

## Implementing the CustomWrapper

The CustomWrapper entity defines the interface that custom designs need to implement, as well as a simple abstraction from the instrument slot. Note when using Verilog, the wrapper needs to be declared before proceeding to define the custom architecture. For ease of use, this wrapper declaration is automatically generated whenever a new Verilog file is created.

:::danger DO NOT EDIT WRAPPER
In the Verilog CustomWrapper, it is recommended to not edit the module declaration. If the module definition is changed, a bitstream may be generated. Deploying the bitstream can potentially damage the Moku device. Please include your custom code in the provided space.
:::

Implementing the CustomWrapper interface simply requires defining an architecture.

<code-group>

<code-block title="VHDL">

```vhdl
architecture Behavioural of CustomWrapper is
begin
    -- your code goes here
end architecture;
```

</code-block>

<code-block title="Verilog">

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

  // your code goes here

endmodule
```

</code-block>

</code-group>

## CustomWrapper Architecture

::: warning
Only one architecture should implement CustomWrapper per project. If multiple
architectures exist, the one that is synthesized is undefined.
:::

<code-group>

<code-block title="VHDL">

```vhdl
entity CustomWrapper is
    port (
        Clk : in std_logic;
        Reset : in std_logic;

        -- Input and Output use is platform-specific. These ports exist on all
        -- platforms but may not be externally connected.
        InputA : in signed(15 downto 0);
        InputB : in signed(15 downto 0);
        InputC : in signed(15 downto 0);
        InputD : in signed(15 downto 0);

        ExtTrig : in std_logic;

        OutputA : out signed(15 downto 0);
        OutputB : out signed(15 downto 0);
        OutputC : out signed(15 downto 0);
        OutputD : out signed(15 downto 0);

        Control0 : in std_logic_vector(31 downto 0);
        Control1 : in std_logic_vector(31 downto 0);
        Control2 : in std_logic_vector(31 downto 0);
        Control3 : in std_logic_vector(31 downto 0);
        Control4 : in std_logic_vector(31 downto 0);
        Control5 : in std_logic_vector(31 downto 0);
        Control6 : in std_logic_vector(31 downto 0);
        Control7 : in std_logic_vector(31 downto 0);
        Control8 : in std_logic_vector(31 downto 0);
        Control9 : in std_logic_vector(31 downto 0);
        Control10 : in std_logic_vector(31 downto 0);
        Control11 : in std_logic_vector(31 downto 0);
        Control12 : in std_logic_vector(31 downto 0);
        Control13 : in std_logic_vector(31 downto 0);
        Control14 : in std_logic_vector(31 downto 0);
        Control15 : in std_logic_vector(31 downto 0)
    );
end entity;
```

</code-block>

<code-block title="Verilog">

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
endmodule
```

</code-block>

</code-group>

## Wrapper Ports

The details of input, output and clock use is platform specific. For details, see [Input and Output](../mcc/io.md).

## Control Registers

These provide control of custom designs at runtime.
See [Control Registers](../mcc/controls.md).
