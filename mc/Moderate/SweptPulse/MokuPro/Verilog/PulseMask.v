module PulseMask (
  input wire clk,				// Clock input
  input wire signed [15:0] passthrough,	    	// Signal to passthrough on mask high 
  input wire unsigned [31:0] divider,		      // Clock divider to control mask frequency
  input wire unsigned [31:0] duty,		        // Controls pulse width of the mask
  output wire signed [15:0] finalOut,		      // Output signal
  output wire signed [15:0] maskDAC		        // Mask representation output for analog port
);

  reg unsigned [31:0] count=32'd0;
  reg unsigned [15:0] r_finalOut, r_maskDAC;

  always @(posedge clk) begin
    if (divider==32'd0 | duty ==32'd0) begin 	// divider or duty of 0 (i.e. initial Control values) will result in a reset condition
      count<=32'd0;
      r_finalOut<=16'd0;			                // mask is 'false', therefore set output to zero
      r_maskDAC<=16'h8000;		              	// Output Largest negative 16 bit number for pulse mask visualization
    end else if (duty>divider) begin		      // if duty cycle is higher than pulse divider, force output high at all times
      count<=32'd0;
      r_finalOut<=passthrough;			          // mask is 'true', therefore passthrough the input signal
      r_maskDAC<=16'h7fff;			              // Largest positive 16 bit number for pulse mask visualization
    end else if (count>=divider-1) begin
      count<=32'd0;
      r_finalOut<=passthrough;			          // mask is 'true', therefore passthrough the input signal
      r_maskDAC<=16'h7fff;      		          // Largest positive 16 bit number for pulse mask visualization
    end else if (count>=duty-1) begin
      count<=count+32'd1;
      r_finalOut<=16'd0;			                // mask is 'false', therefore set output to zero
      r_maskDAC<=16'h8000;			              // Output Largest negative 16 bit number for pulse mask visualization
    end else begin
      r_finalOut<=passthrough;
      count<=count+1;
    end 
  end
  
  assign finalOut = r_finalOut;
  assign maskDAC = r_maskDAC;
  
endmodule
