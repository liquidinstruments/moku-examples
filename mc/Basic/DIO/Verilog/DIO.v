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

  
// To use this, you must configure the MCC block in the Multi-instrument Mode builder as follows:
// MCC Slot's Input A -> DIO
// MCC Slot's Output B -> DIO
// DIO Pin 1-8 set as Input
// DIO Pin 9-16 set as Output

  reg [2:0] Count;

  assign OutputA[0] = InputA[8]; 			// Loop back Pin 9 to Pin 1
  assign OutputA[1] = !InputA[9]; 		// Pin 2 is the inverse of Pin 10
  assign OutputA[2] = Count[0]; 			// Pin 3 is a clock at 15.625MHz (Moku:Go MCC core clock is 31.25MHz)				
  assign OutputA[3] = Count[1]; 			// Pin 4 is a clock at half the rate of Pin 3
  assign OutputA[4] = Count[2];				// and Pin 5 is half the rate again

  assign OutputA[5] = InputA[10] & InputA[11]; 		// Logical AND
  assign OutputA[6] = InputA[10] | InputA[11];		// Logical OR

  always @(posedge Clk) begin
    if (Reset == 1'b1)
      Count <= 3'b000;
    else
      Count <= Count+ 3'b001;
  end

endmodule
