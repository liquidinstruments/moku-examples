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
  
  MovingAverage #(.G_AVERAGE_LENGTH_LOG(10)) M1(clk, reset, inputa, inputb, c1][15:0], control[2][15:0], outputa);
  MovingMedian M2(clk, reset, inputa, outputb);
  
endmodule
