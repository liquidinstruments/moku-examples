module CustomInstrumentInterlaced 
#(  parameter input_interlacing_factor,
    parameter output_interlacing_factor
)(
    input wire clk,
    input wire reset,
    input wire [31:0] sync,

    input wire signed [15:0] inputa [0:input_interlacing_factor - 1],
    input wire signed [15:0] inputb [0:input_interlacing_factor - 1],
    input wire signed [15:0] inputc [0:input_interlacing_factor - 1],
    input wire signed [15:0] inputd [0:input_interlacing_factor - 1],

    input wire exttrig,

    output wire signed [15:0] outputa [0:output_interlacing_factor - 1],
    output wire signed [15:0] outputb [0:output_interlacing_factor - 1],
    output wire signed [15:0] outputc [0:output_interlacing_factor - 1],
    output wire signed [15:0] outputd [0:output_interlacing_factor - 1],

    input wire [31:0] control [0:15],
    output wire [31:0] status[0:15]
);

// genvar k;
// generate
//      for (k=0; k < input_interlacing_factor ; k = k + 1)
//          begin
//              _________ <= inputa[k];
//              _________ <= inputb[k];
//              _________ <= inputc[k];
//              _________ <= inputd[k];
//          end
// endgenerate

// assign ______ = control[0];
// assign ______ = control[1];
// assign ______ = control[2];
//        ......
// assign ______ = control[15];

// generate
//      for (k=0; k < output_interlacing_factor ; k = k + 1)
//          begin
//              assign outputa[k] = ______;
//              assign outputb[k] = ______;
//              assign outputc[k] = ______;
//              assign outputd[k] = ______;
//          end
// endgenerate

// assign status[0] = ______;
// assign status[1] = ______;
// assign status[2] = ______;
          ......
// assign status[15] = ______;

endmodule
