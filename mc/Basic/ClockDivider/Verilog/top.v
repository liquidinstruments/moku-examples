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

// Designed by Brian J. Neff / Liquid Instruments
// Will produce a clock divider and output the divided clock to specified pin
// Moku:Go should be configured as follows:
// DIO Pin 0 to Input - Will reset the system on logical True
// DIO Pin 8 to Output - Will output the divided clock pulse by a factor of 2
// DIO Pin 9 to Output - Will output the divided clock pulse by a factor of 4
// DIO Pin 10 to Output - Will output the divided clock pulse by a factor of 6
// All other pins remain unused and can be configured as input or output

  clkdiv u_ClkDivider1(
   .clk(Clk),
   .reset(InputA[0]),
   .pulse(OutputA[8]));

// Create additional entities to highlight value of using parameter 

  clkdiv #(.divider(2)) u_ClkDivider2(
   .clk(Clk),
   .reset(InputA[0]),
   .pulse(OutputA[9]));
  
  clkdiv #(.divider(3)) u_ClkDivider3(
   .clk(Clk),
   .reset(InputA[0]),
   .pulse(OutputA[10]));

endmodule
