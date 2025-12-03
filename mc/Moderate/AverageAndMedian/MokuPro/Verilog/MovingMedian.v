module MovingMedian (
  input wire Clk,                           // Clock signal
  input wire Reset,                         // Synchronous reset
  input wire signed [15:0] Input,           // Input (signed 16-bit)
  output wire signed [15:0] Output          // Output median value (signed 16-bit)
);
// Holds the last 5 input values
  reg signed [15:0] moving_window [0:4];

// Multi-stage sorting pipeline: 6 stages, each with 5 values flattened into 1D array of 30 elements
// 1D array contains 6 pipelines, each starting at multiples of 5. Elements [0 to 4] represent first stage, 
// Elements [5:9] represent second stage, Elements [10:14] represent third stage and so on.
  reg signed [15:0] staged_sort [0:29];
  
// Holds the final median value after all sorting stages
  reg signed [15:0] median;

///////////////////////////////////////////////
// Shift in new data into moving window buffer
///////////////////////////////////////////////
  always @(posedge Clk) begin
    integer i;
    if (Reset==1'b1)
      for (i=0; i<4; i=i+1) begin
        moving_window[i] <= 0;                        // Clear window on reset
      end
    else
      moving_window <= {Input, moving_window[0:3]};   // Shift and insert new sample
  end

////////////////////////////////////////////////////////
// Pipelined sorting network to compute median value
////////////////////////////////////////////////////////
  always @(posedge Clk) begin
    integer i,j;
    if (Reset==1'b1) begin
      for (i=0; i<30; i=i+1) begin
          staged_sort[i] <= 0;                        // Clear sorting pipeline
      end
      median <= 16'd0;
    end else begin
    //////////////////////////////////////////////////////////////////////////////
    // Stage 0: Load the current moving window into the first stage
    //////////////////////////////////////////////////////////////////////////////
      for (j=0;j<5; j=j+1) begin
        staged_sort[j] <= moving_window[j];
      end

    ///////////////////////////////////////////////////////////////////////////
    // Stage 1: Compare and sort pairs (0,1) and (2,3)
    ///////////////////////////////////////////////////////////////////////////
      if (staged_sort[0]>staged_sort[1]) begin
        staged_sort[5] <= staged_sort[1];
        staged_sort[6] <= staged_sort[0];
      end else begin
        staged_sort[5] <= staged_sort[0];
        staged_sort[6] <= staged_sort[1];
      end
      if (staged_sort[2]>staged_sort[3]) begin
        staged_sort[7] <= staged_sort[3];
        staged_sort[8] <= staged_sort[2];
      end else begin
        staged_sort[7] <= staged_sort[2];
        staged_sort[8] <= staged_sort[3];
      end
      staged_sort[9]<=staged_sort[4]; // Element 4 remains unprocessed in this stage   

    ///////////////////////////////////////////////////////////////////////////
    // Stage 2: Compare (1,2) and (3,4), pass 0 through unchanged
    ///////////////////////////////////////////////////////////////////////////
      staged_sort[10]<= staged_sort[5];
      if (staged_sort[6]>staged_sort[7]) begin
        staged_sort[11] <= staged_sort[7];
        staged_sort[12] <= staged_sort[6];
      end else begin
        staged_sort[11] <= staged_sort[6];
        staged_sort[12] <= staged_sort[7];
      end
      if (staged_sort[8]>staged_sort[9]) begin
        staged_sort[13] <= staged_sort[9];
        staged_sort[14] <= staged_sort[8];
      end else begin
        staged_sort[13] <= staged_sort[8];
        staged_sort[14] <= staged_sort[9];
      end

    ///////////////////////////////////////////////////////////////////////////
    // Stage 3: Compare (0,1) and (2,3), pass 4 through
    ///////////////////////////////////////////////////////////////////////////
      if (staged_sort[10]>staged_sort[11]) begin
        staged_sort[15] <= staged_sort[11];
        staged_sort[16] <= staged_sort[10];
      end else begin
        staged_sort[15] <= staged_sort[10];
        staged_sort[16] <= staged_sort[11];
      end
      if (staged_sort[12]>staged_sort[13]) begin
        staged_sort[17] <= staged_sort[13];
        staged_sort[18] <= staged_sort[12];
      end else begin
        staged_sort[17] <= staged_sort[12];
        staged_sort[18] <= staged_sort[13];
      end
      staged_sort[19] <= staged_sort[14];

    ///////////////////////////////////////////////////////////////////////////
    //  Stage 4: Compare (1,2) and (3,4), pass 0 through
    ///////////////////////////////////////////////////////////////////////////
      staged_sort[20] <= staged_sort[20];
      if (staged_sort[16]>staged_sort[17]) begin
        staged_sort[21] <= staged_sort[17];
        staged_sort[22] <= staged_sort[16];
      end else begin
        staged_sort[21] <= staged_sort[16];
        staged_sort[22] <= staged_sort[17];
      end
      if (staged_sort[18]>staged_sort[19]) begin
        staged_sort[23] <= staged_sort[19];
        staged_sort[24] <= staged_sort[18];
      end else begin
        staged_sort[23] <= staged_sort[18];
        staged_sort[24] <= staged_sort[19];
      end

    ///////////////////////////////////////////////////////////////////////////
    //  Stage 5: Final sort pass (0,1) and (2,3), pass 4 through
    ///////////////////////////////////////////////////////////////////////////
      if (staged_sort[20]>staged_sort[21]) begin
        staged_sort[25] <= staged_sort[21];
        staged_sort[26] <= staged_sort[20];
      end else begin
        staged_sort[25] <= staged_sort[20];
        staged_sort[26] <= staged_sort[21];
      end
      if (staged_sort[22]>staged_sort[23]) begin
        staged_sort[27] <= staged_sort[23];
        staged_sort[28] <= staged_sort[22];
      end else begin
        staged_sort[27] <= staged_sort[22];
        staged_sort[28] <= staged_sort[23];
      end
      staged_sort[29] <= staged_sort[24];

    ////////////////////////////////////////////////////////////////////////////
    // After all sorting stages, pick the median value (middle of sorted 5)
    ////////////////////////////////////////////////////////////////////////////
      median <=staged_sort[27];
    end
  end

// Output the median value
  assign Output = median;

endmodule
