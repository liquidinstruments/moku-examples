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

  
// To use this, you must configure the MCC block in the Multi-instrument Mode builder as follows:
// MCC Slot's Input A -> DIO
// MCC Slot's Output B -> DIO
// DIO Pin 1-8 set as Input
// DIO Pin 9-16 set as Output

  reg [2:0] Count;

  assign outputa[0] = inputa[8]; 			// Loop back Pin 9 to Pin 1
  assign outputa[1] = !inputa[9]; 		// Pin 2 is the inverse of Pin 10
  assign outputa[2] = Count[0]; 			// Pin 3 is a clock at 15.625MHz (Moku:Go MCC core clock is 31.25MHz)				
  assign outputa[3] = Count[1]; 			// Pin 4 is a clock at half the rate of Pin 3
  assign outputa[4] = Count[2];				// and Pin 5 is half the rate again

  assign outputa[5] = inputa[10] & inputa[11]; 		// Logical AND
  assign outputa[6] = inputa[10] | inputa[11];		// Logical OR

  always @(posedge clk) begin
    if (reset == 1'b1)
      Count <= 3'b000;
    else
      Count <= Count+ 3'b001;
  end

endmodule
