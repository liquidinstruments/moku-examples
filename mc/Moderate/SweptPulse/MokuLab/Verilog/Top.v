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

// Designed by Brian J. Neff / Liquid Instruments
// Will use the devices internal clock along with Control Register inputs to create a variable frequency and pulse width output
// This code will work as is on any version of the Moku.  However, there is an alternate version that also uses the DIO for the Moku:Go
// Moku should be configured as follows:
// Control0 register must be non-zero integer
// Control1 register must be non-zero integer 

PulseMask P1(
    .clk(Clk),			 
    .passthrough(InputB), 		// Will pass this signal through to output when mask is high
    .divider(Control0[31:0]), 	// Output pulse divider to control frequency
    .duty(Control1[31:0]),		// Sets the duty cycle of the output pulse
    .finalOut(OutputA), 		// Either 0 (when Mask is 0) or InputB (when Mask is 1)
    .maskDAC(OutputB) 		    // Mask representation output to DAC linked to OutputC in Multi-instrument Mode
);  

endmodule
