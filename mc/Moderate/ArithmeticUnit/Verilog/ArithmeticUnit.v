module ArithmeticUnit (
  input wire signed [15:0] A,        // Signed 16-bit input A
  input wire signed [15:0] B,        // Signed 16-bit input B
  input wire [1:0] OpCode,           // 2-bit operation selector
  output wire signed [15:0] Result,  // Signed 16-bit result output
  input wire Clk                     // Clock signal for synchronous operation
);

// Intermediate wires and register
  wire signed [15:0] Add, Sub;       // Wires to hold addition and subtraction results
  wire signed [31:0] Mult;           // Wire to hold 32-bit multiplication result
  reg signed [15:0] temp;            // Register to store the selected operation result

// Assign arithmetic operations
  assign Add = A + B;                // Addition
  assign Sub = A - B;                // Subtraction
  assign Mult = A * B;               // Multiplication

// Synchronous logic block (occurs once every clock cycle) that selects operation based on OpCode
  always @(posedge Clk) begin
    case (OpCode)
      2'b00: temp = Add;            // OpCode 00: Assign addition result
      2'b01: temp = Sub;            // OpCode 01: Assign subtraction result
      2'b10: temp = Mult[31:16];    // OpCode 10: Assign upper 16 bits of multiplication result
      default: temp = A;            // OpCode 11 or others: Default passing input A to output A
    endcase
  end

// Assign the calculated temp value to the output (Result)
  assign Result = temp;

endmodule
