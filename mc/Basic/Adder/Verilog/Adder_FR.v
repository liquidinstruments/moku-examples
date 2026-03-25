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
  
// Assign sum of inputs to OutputA and difference of inputs to OutputB

genvar k;
generate
     for (k=0; k < input_interlacing_factor ; k = k + 1)
         begin
            // Assign sum of inputs A and B to OutputA
            assign outputa[k] = inputa[k] + inputb[k];
            
            // Assign difference of inputs A and B to OutputB
            assign outputb[k] = inputa[k] - inputb[k];
         end
endgenerate

endmodule