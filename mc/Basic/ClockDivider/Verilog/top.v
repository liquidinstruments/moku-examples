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

    output wire outputinterpa,
    output wire outputinterpb,
    output wire outputinterpc,
    output wire outputinterpd,

    input wire [31:0] control [0:15],
    output wire [31:0] status[0:15]
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
   .clk(clk),
   .reset(inputa[0]),
   .pulse(outputa[8]));

// Create additional entities to highlight value of using parameter 

  clkdiv #(.divider(2)) u_ClkDivider2(
   .clk(clk),
   .reset(inputa[0]),
   .pulse(outputa[9]));
  
  clkdiv #(.divider(3)) u_ClkDivider3(
   .clk(clk),
   .reset(inputa[0]),
   .pulse(outputa[10]));

endmodule
