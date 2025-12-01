////////////////////////////////////////////////////////////////
// Implementation of a DC sequencer
// Sequence of DC levels are generated on DataOutA,
// new level each time a trigger event occurs on DataIn.
// Trigger has a Schmitt style trigger with a lo and hi
// threshold. The cleaned trigger signal is output on
// DataOutB for use downstream.
// Configurable parameters:
//    Schmitt trigger high value (bits) = Control0[31:16]
//    Schmitt trigger low value (bits) = Control0[15:0]
//    Trigger output high and low values, constants
//    DC value sequence, array of constants
////////////////////////////////////////////////////////////////

module DCSequencer (
  input wire Clk,                      // System clock
  input wire Reset,                    // Synchronous reset
  input wire signed [15:0] DataIn,     // Input signal to monitor
  input wire signed [15:0] HIThreshold,// High threshold level
  input wire signed [15:0] LOThreshold,// Low threshold level
  output wire signed [15:0] DataOutA,  // Sequenced DC output
  output wire signed [15:0] DataOutB   // Indicates trigger status (HI or LO level)
);

// Predefined lookup table with 128 signed 16-bit constants
  reg signed [15:0] DC [0:127] = {
        16'h371E, 16'hF110, 16'hF607, 16'h53A3, 16'h120E, 16'hF9EE, 16'hF2AD, 16'hF968,
        16'hF6D1, 16'hD494, 16'hF5B8, 16'h0F78, 16'h0D18, 16'hDB9B, 16'hEDAA, 16'hE75A,
        16'hEFF4, 16'h3BC6, 16'hE4EE, 16'hF99A, 16'h072A, 16'hF5D8, 16'h11CB, 16'h268F,
        16'h27C1, 16'hE3C9, 16'h1F00, 16'hED97, 16'hEBD7, 16'h20BA, 16'hDB56, 16'hF647,
        16'hEA9B, 16'hF238, 16'h191F, 16'hD38C, 16'hDD0B, 16'hDC62, 16'hDD81, 16'hCBB7,
        16'h3777, 16'hDE71, 16'h0500, 16'h2AB7, 16'h0F40, 16'hBB1D, 16'hEAF6, 16'hFFAE,
        16'hF9E2, 16'h21B3, 16'hEEB1, 16'hAFD1, 16'hF48A, 16'h1069, 16'h1776, 16'h1315,
        16'hCEE0, 16'h9D46, 16'hDEC7, 16'hDA28, 16'h05D2, 16'h4558, 16'hDFBA, 16'h4568,
        16'hE9B1, 16'hF616, 16'hE688, 16'h182C, 16'h20DF, 16'hF6D8, 16'h09F8, 16'hD34D,
        16'h24AD, 16'h4423, 16'hA9D3, 16'hE97F, 16'h2F27, 16'h0FA7, 16'hC64B, 16'hD25F,
        16'h5DA9, 16'hF286, 16'h2591, 16'hF354, 16'h0CC9, 16'hB523, 16'hE7CC, 16'h106C,
        16'hF2A3, 16'h16C7, 16'h24BD, 16'hEBB2, 16'hC42E, 16'h2008, 16'hFE75, 16'hF92C,
        16'hD44F, 16'h24AC, 16'h0B6A, 16'hEC3E, 16'hF26E, 16'hF0E8, 16'hEB86, 16'hEEC3,
        16'h25D1, 16'hF824, 16'hF31C, 16'h007B, 16'hE1B2, 16'h1AD9, 16'h285F, 16'hE38F,
        16'hC6C3, 16'h0E85, 16'h2616, 16'h20B5, 16'h44AD, 16'h45F7, 16'h02FD, 16'hC927,
        16'h1E7B, 16'hB58E, 16'h2270, 16'h23B5, 16'hC042, 16'h933F, 16'h044A, 16'h1E82
  };

// Constants to drive output DataOutB depending on trigger state
  reg signed [15:0] HI_LVL = 16'h7FFF; // High-level output value
  reg signed [15:0] LO_LVL = 16'h0000; // Low-level output value

// Internal registers
  reg Step;             // One-clock pulse when DataIn crosses the HI threshold
  reg Trigger;          // Current trigger state 
  reg TriggerDly;       // Delayed version of Trigger to detect rising edge
  reg signed [15:0] temp_data;  // Holds the current DC value from LUT
  reg unsigned [6:0] DCLevelAddr; // Address pointer into DC LUT
  integer int_data;     // Used for addressing the array

// Edge detection logic for triggering
  always @(posedge Clk) begin
    if (Reset == 1'b1)
      Trigger <= 1'b0;
    else if (DataIn >= HIThreshold)
      Trigger <= 1'b1; // Trigger goes high when DataIn exceeds HI threshold
    else if (DataIn < LOThreshold)
      Trigger <= 1'b0; // Trigger goes low when DataIn falls below LO threshold

    TriggerDly <= Trigger;            // Store previous value of Trigger
    Step <= Trigger & (~TriggerDly);  // Generate single pulse when Trigger rises
  end

// Increment address pointer when a rising edge is detected
  always @(posedge Clk) begin
    if (Reset == 1'b1)
      DCLevelAddr <= 0;                  // Reset address pointer
    else if (Step == 1'b1)
      DCLevelAddr <= DCLevelAddr + 7'd1; // Increment on rising edge of Trigger
  end

// Read current DC value from LUT using DCLevelAddr
  always @(posedge Clk) begin
    int_data = DCLevelAddr;           // Assign address to integer for indexing
    temp_data <= DC[int_data];        // Lookup corresponding DC value
  end

// Output assignments
  assign DataOutA = temp_data;              // Output from the DC lookup table
  assign DataOutB = (Trigger == 1'b1) ?     // Output high or low level depending on trigger state
                    HI_LVL : LO_LVL;

endmodule
