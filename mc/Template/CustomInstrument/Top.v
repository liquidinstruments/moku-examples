module CustomInstrument (
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

    input wire [31:0] control [0:15],
    output wire [31:0] status [0:15]
);

// _________ <= InputA;
// _________ <= InputB;
// _________ <= InputC;
// _________ <= InputD;

// assign ______ = control[0];
// assign ______ = control[1];
// assign ______ = control[2];
//        ......
// assign ______ = control[15];


// assign OutputA = ______;
// assign OutputB = ______;
// assign OutputC = ______;
// assign OutputD = ______;

// assign status[0] = ______;
// assign status[1] = ______;
// assign status[2] = ______;
          ......
// assign status[15] = ______;
endmodule
