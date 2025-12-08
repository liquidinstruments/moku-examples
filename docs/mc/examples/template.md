# FPGA coding template

Provided is a blank template for creating HDL files using the Moku Compile either using CustomWrapper or CustomInstrument entity

## CustomInstrument Template

| Port                                                                  | In/Out | Type             | Range       |
| --------------------------------------------------------------------- | ------ | ---------------- | ----------- |
| Clk                                                                   | in     | std_logic        | -           |
| Reset                                                                 | in     | std_logic        | -           |
|                                                                       |        |                  |             |
| InputA                                                                | in     | signed           | 15 downto 0 |
| InputB                                                                | in     | signed           | 15 downto 0 |
| InputC <small><br> (Moku:Pro only) </br></small>                      | in     | signed           | 15 downto 0 |
| InputD <small><br> (Moku:Pro only) </br></small>                      | in     | signed           | 15 downto 0 |
|                                                                       |        |                  |             |
| ExtTrig <small><br> (Moku:Lab, Moku:Pro and Moku:Delta) </br></small> | in     | std_logic        | -           |
|                                                                       |
| OutputA                                                               | out    | signed           | 15 downto 0 |
| OutputB                                                               | out    | signed           | 15 downto 0 |
| OutputC <small><br> (Moku:Go and Moku:Pro) </br></small>              | out    | signed           | 15 downto 0 |
| OutputD <small><br> (Moku:Pro only) </br></small>                     | out    | signed           | 15 downto 0 |
|                                                                       |        |                  |             |
| Control                                                               | in     | array_of_slv     | 0 to 15     |
|                                                                       |        |                  |             |
| Status                                                                | out    | array_of_slv     | 0 to 15     |
|                                                                       |        |                  |             |

<code-group>

<code-block title="CustomInstrument VHDL">

<<< @/docs/api/moku-examples/mc/Template/CustomInstrument/Top.vhd

</code-block>

<code-block title="CustomInstrument Verilog">

<!-- Update when moku-examples changes -->
<!-- <<< @/docs/api/moku-examples/mc/Template/CustomInstrument/Top.v -->
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

## CustomWrapper Template

| Port                                                                  | In/Out | Type             | Range       |
| --------------------------------------------------------------------- | ------ | ---------------- | ----------- |
| Clk                                                                   | in     | std_logic        | -           |
| Reset                                                                 | in     | std_logic        | -           |
|                                                                       |        |                  |             |
| InputA                                                                | in     | signed           | 15 downto 0 |
| InputB                                                                | in     | signed           | 15 downto 0 |
| InputC <small><br> (Moku:Pro only) </br></small>                      | in     | signed           | 15 downto 0 |
| InputD <small><br> (Moku:Pro only) </br></small>                      | in     | signed           | 15 downto 0 |
|                                                                       |        |                  |             |
| ExtTrig <small><br> (Moku:Lab, Moku:Pro and Moku:Delta) </br></small> | in     | std_logic        | -           |
|                                                                       |
| OutputA                                                               | out    | signed           | 15 downto 0 |
| OutputB                                                               | out    | signed           | 15 downto 0 |
| OutputC <small><br> (Moku:Go and Moku:Pro) </br></small>              | out    | signed           | 15 downto 0 |
| OutputD <small><br> (Moku:Pro only) </br></small>                     | out    | signed           | 15 downto 0 |
|                                                                       |        |                  |             |
| Control0                                                              | in     | std_logic_vector | 31 downto 0 |
| Control1                                                              | in     | std_logic_vector | 31 downto 0 |
| ...                                                                   | ...    | ...              | ...         |
| Control15                                                             | in     | std_logic_vector | 31 downto 0 |
|                                                                       |        |                  |             |

<code-group>

<code-block title="CustomWrapper VHDL">

<<< @/docs/api/moku-examples/mc/Template/CustomWrapper/Top.vhd

</code-block>

<code-block title="CustomWrapper Verilog">

<<< @/docs/api/moku-examples/mc/Template/CustomWrapper/Top.v

</code-block>

</code-group>
