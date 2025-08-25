# VHDL coding template

Provided is a blank template for creating HDL files using the Moku Cloud Compile.

## Entity Ports

| Port                                                                  | In/Out | Type             | Range       |
| --------------------------------------------------------------------- | ------ | ---------------- | ----------- |
| Clk                                                                   | in     | std_logic        | -           |
| Reset                                                                 | in     | std_logic        | -           |
|                                                                       |        |                  |             |
| InputA                                                                | in     | signed           | 15 downto 0 |
| InputB                                                                | in     | signed           | 15 downto 0 |
| InputC <small><br> (Moku:Pro and Moku:Delta) </br></small>            | in     | signed           | 15 downto 0 |
| InputD <small><br> (Moku:Pro and Moku:Delta) </br></small>            | in     | signed           | 15 downto 0 |
| InputE <small><br> (Moku:Delta only) </br></small>                    | in     | signed           | 15 downto 0 |
| InputF <small><br> (Moku:Delta only) </br></small>                    | in     | signed           | 15 downto 0 |
| InputG <small><br> (Moku:Delta only) </br></small>                    | in     | signed           | 15 downto 0 |
| InputH <small><br> (Moku:Delta only) </br></small>                    | in     | signed           | 15 downto 0 |
                                                                        |        |                  |             |
| ExtTrig <small><br> (Moku:Lab, Moku:Pro and Moku:Delta) </br></small> | in     | std_logic        | -           |
|                                                                       |
| OutputA                                                               | out    | signed           | 15 downto 0 |
| OutputB                                                               | out    | signed           | 15 downto 0 |
| OutputC <small><br> (Moku:Go, Moku:Pro, and Moku:Delta) </br></small> | out    | signed           | 15 downto 0 |
| OutputD <small><br> (Moku:Pro and Moku:Delta) </br></small>           | out    | signed           | 15 downto 0 |
| OutputE <small><br> (Moku:Delta only) </br></small>                   | out    | signed           | 15 downto 0 |
| OutputF <small><br> (Moku:Delta only) </br></small>                   | out    | signed           | 15 downto 0 |
| OutputG <small><br> (Moku:Delta only) </br></small>                   | out    | signed           | 15 downto 0 |
| OutputH <small><br> (Moku:Delta only) </br></small>                   | out    | signed           | 15 downto 0 |
|                                                                       |        |                  |             |
| Control0                                                              | in     | std_logic_vector | 31 downto 0 |
| Control1                                                              | in     | std_logic_vector | 31 downto 0 |
| ...                                                                   | ...    | ...              | ...         |
| Control15                                                             | in     | std_logic_vector | 31 downto 0 |
|                                                                       |        |                  |             |

<code-group>

<code-block title="VHDL">

<<< @/docs/api/moku-examples/mcc/Template/Top.vhd

</code-block>

<code-block title="Verilog">

<<< @/docs/api/moku-examples/mcc/Template/Top.v

</code-block>

</code-group>
