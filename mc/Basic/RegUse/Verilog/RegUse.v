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

// Convert values in control registers to voltage levels on the outputs
// The 16 bit scale covers the peak-to-peak range of the device
// Control values 1 - 32767 output voltages up to V_peak
// Control values 32768 - 65536 output voltages from -V_peak to 0 V

  assign outputa = control[1][15:0];
  assign outputb = control[2][15:0];

endmodule
