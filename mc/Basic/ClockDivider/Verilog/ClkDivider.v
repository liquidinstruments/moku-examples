// Designed by Brian J. Neff / Liquid Instruments
// This component will produce a clock divider 
// The divider signal below can be adjusted to specify how many times you wish to divide the clock

module clkdiv #(parameter divider=1)(
  input wire clk,                               // Clock input
  input wire reset,                             // Reset input
  output reg pulse                              // Output pulse
);

  reg [15:0] count;                             // 16-bit counter register

  always @(posedge clk) begin
    if (reset) begin                            
      count = 16'h0000;                   // Reset the counter
      pulse = 1'b0;                       // Reset the pulse output
    end else begin
      if (count >= (divider-1)) begin     // When counter reaches divider value
          pulse = !pulse;                 // Toggle output pulse
          count = 16'h0000;               // Reset counter
      end else
          count = count + 16'h0001;       // Otherwise, increment the counter
    end
  end
  
endmodule
