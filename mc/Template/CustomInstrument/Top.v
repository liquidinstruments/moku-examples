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

// _________ <= inputa;
// _________ <= inputb;
// _________ <= inputc;
// _________ <= inputd;

// assign ______ = control[0];
// assign ______ = control[1];
// assign ______ = control[2];
//        ......
// assign ______ = control[15];


// assign outputa = ______;
// assign outputb = ______;
// assign outputc = ______;
// assign outputd = ______;

// assign status[0] = ______;
// assign status[1] = ______;
// assign status[2] = ______;
          ......
// assign status[15] = ______;

endmodule
