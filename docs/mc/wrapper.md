# Wrapper for Moku Compile

:::danger Wrapper changes
From MokuOS version 4.1.1 and above, we recommend users to use CustomInstrument entity wrapper. This change mainly affect the Control register definition and the availability of Status registers. For legacy purposes, CustomWrapper is still available but will be deprecated soon.
:::

To implement your custom design, Moku Compile requires an entity to define the interface as well as a simple abstraction from the instrument slot. The wrapper is provided in both VHDL and Verilog (SystemVerilog).

Note when using Verilog, the wrapper needs to be declared before proceeding to define the custom architecture. For ease of use, this wrapper declaration is automatically generated whenever a new Verilog file is created.

## Implementing the wrapper
This wrapper is implemented either as CustomInstrument or CustomWrapper (for MokuOS versions older than 4.1.1). Implementing the wrapper interface simply requires defining an architecture using either CustomInstrument or CustomWrapper.

:::warning Wrapper Implementation
1. Only one architecture should exist in your project, using either the CustomWrapper or CustomInstrument entity. If multiple architectures exist, the one that is synthesized could be potentially undefined.
2. It is recommended not to modify the Verilog module declaration. While a bitstream may still be generated if the module definition is changed, it might not behave as expected.
:::

## CustomInstrument Architecture
For Moku devices running MokuOS versions 4.1.1 and newer, the CustomInstrument entity can be used to define the architecture. This entity allows the use of status registers for the custom designs.

<code-group>

<code-block title="VHDL">

```vhdl
entity CustomInstrument is
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

        Control : in array_of_slv(0 to 15);
        Status  : in array_of_slv(0 to 15);
    );
end entity;
```

</code-block>

<code-block title="Verilog">

```verilog
module CustomInstrument (
    input wire clk,
    input wire reset,
    input wire [31:0] sync,

    input wire signed [15:0] inputa,
    input wire signed [15:0] inputb,
    input wire signed [15:0] inputc,
    input wire signed [15:0] inputd,

    input wire exttrig,

    output wire signed [15:0] outputa,
    output wire signed [15:0] outputb,
    output wire signed [15:0] outputc,
    output wire signed [15:0] outputd,

    input wire [31:0] control [0:15],
    output wire [31:0] status[0:15]
);
endmodule
```

</code-block>

</code-group>

## CustomWrapper Architecture

For Moku device running MokuOS versions 4.0.3 and older, they require the legacy CustomWrapper to be defined in the architecture. Note that this wrapper architecture will be deprecated in future. While it is currently supported, we recommend updating projects to implement the CustomInstrument.

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

The details of input, output and clock use is platform specific. For details, see [input and output](../mc/io.md).

## Control Registers

These provide control of custom designs at runtime.
See [control registers](../mc/controls.md).

## Status Registers

These registers can be used as indicators in custom designs during runtime.
See [status registers](../mc/statusregs.md).