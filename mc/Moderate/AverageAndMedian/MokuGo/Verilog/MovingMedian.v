module MovingMedian (
  input wire Clk,
  input wire Reset,
  input wire signed [15:0] Input,
  output wire signed [15:0] Output  
);

  reg signed [15:0] moving_window [0:4];
  reg signed [15:0] sorted_window [0:4];
  reg signed [15:0] median;

  always @(posedge Clk) begin
    integer j;
    if (Reset==1'b1)
      for (j=0; j<4; j=j+1) begin
        moving_window[j] <= 16'd0;
      end
    else
      moving_window <= {Input, moving_window[0:3]};
  end

  always @(posedge Clk) begin
    reg signed [15:0] temp;
    integer i,j,k;
    reg signed [15:0] var_array [0:4];
    
    var_array <= moving_window;
    if (Reset==1'b1) begin
      for (k=0; k<4; k=k+1) begin
        sorted_window[k] <= 16'd0;
      end
      median <= 16'd0;
    end
    else begin
      for (j=0; j<4; j=j+1) begin
        for (i=0; i<(4-j); i=i+1) begin
          if (var_array[i] > var_array[i+1]) begin
            temp=var_array[i];
            var_array[i]=var_array[i+1];
            var_array[i+1]=temp;
          end
        end
      end
      sorted_window <=var_array;
      median<=sorted_window[2];
    end
  end

assign Output = median;

endmodule
