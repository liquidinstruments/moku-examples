# VHDL coding template

Provided is a blank template for creating HDL files using the Moku Cloud Compile.

## Entity Ports

| Port | In/Out | Type | Range |
| ----- | ----- | ----- | ----- |
| Clk | in | std_logic | - |
| Reset | in | std_logic | - |
||
| InputA | in | signed | 15 downto 0 |
| InputB | in | signed | 15 downto 0 |
| InputC <small><br> (Moku:Pro only) </br></small>| in | signed | 15 downto 0 |
| InputD <small><br> (Moku:Pro only) </br></small> | in | signed | 15 downto 0 |
||
| ExtTrig <small><br> (Moku:Lab and Moku:Pro) </br></small>| in | std_logic | - |
||
| OutputA | out | signed | 15 downto 0 |
| OutputB | out | signed | 15 downto 0 |
| OutputC <small><br> (Moku:Go and Moku:Pro) </br></small>| out | signed | 15 downto 0 |
| OutputD <small><br> (Moku:Pro only) </br></small>| out | signed | 15 downto 0 |
||
| Control0 | in | std_logic_vector | 31 downto 0 |
| Control1 | in | std_logic_vector | 31 downto 0 |
| ... | ... | ... | ... |
| Control15 | in | std_logic_vector | 31 downto 0 |
||


<code-group>

<code-block title=VHDL>

<<< @/docs/api/moku-examples/mcc/Template/Top.vhd

</code-block>

<code-block title="Verilog">
```
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

// _________ <= InputA;
// _________ <= InputB;
// _________ <= InputC;
// _________ <= InputD;

// assign ______ = Control0;
// assign ______ = Control1;
// assign ______ = Control2;
//        ......
// assign ______ = Control15;


// assign OutputA = ______;
// assign OutputB = ______;
// assign OutputC = ______;
// assign OutputD = ______;
endmodule
```

</code-block>

</code-group>
