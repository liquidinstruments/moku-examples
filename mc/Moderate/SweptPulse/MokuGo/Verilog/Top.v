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
// Designed by Brian J. Neff / Liquid Instruments
// Will use the devices internal clock to create a variable frequency and pulse width output
// Will need to be adjusted for the clock rate of specific device (Moku:Go clock is 31.25 MHz)
// Moku should be configured as follows:
// DIO Pin 0 is input
// DIO Pin 8 is output
// Control[0] register must be non-zero integer
// Control[1] register must be non-zero integer 

  PulseMask P1(
    .clk(clk),			 
    .reset(inputa[0]),		    // Reset input on DIO pin-0
    .passthrough(inputb), 		// Will pass this signal through to output when mask is high
    .divider(control[0][31:0]), // Output pulse divider to control frequency
    .duty(control[1][31:0]),		// Sets the duty cycle of the output pulse
    .finalOut(outputb), 		  // Either 0 (when Mask is 0) or InputB (when Mask is 1)
    .maskDAC(outputc), 		    // Mask representation output to DAC linked to OutputC in Multi-instrument Mode
    .maskDIO(outputa[8])		  // Mask representation output to DIO pin-8 
  );

endmodule
