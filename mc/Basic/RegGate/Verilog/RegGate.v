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
  
// The format below follows the following logic:
// condition ? value_if_true : value_if_false

  assign outputa = control[1][0] ? inputa : 16'h000;  // If the 0th bit of the control is 1, the input is passed to the output.
  assign outputb = control[2][0] ? inputb : 16'h000;  // If the 0th bit of the control is 0, the output is 0.

endmodule
