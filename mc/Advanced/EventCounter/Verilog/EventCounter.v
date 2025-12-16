// Implementation of a simple Event Counter
// counts events of PulseMin >= time > PulseMax (unit clks)
// in periods of PeriodCounter_limit (unit clks)
// If count > MinPulseCount then set OutA HI
// OutB is invert of OutA

// e.g. for pulses from min 48ns to max 99.2ns
// Pulse_min = 0x0f
// Pulse_max = 0x1f
// PeriodCounter_limit = 0x0c35
// MinPulseCount = 0x19

module EventCounter (
  input wire Clk,
  input wire Reset,
  input wire signed [15:0] DataIn,
  input wire unsigned [31:0] PeriodCounterLimit,
  input wire unsigned [15:0] PulseMin,
  input wire unsigned [15:0] PulseMax,
  input wire signed [15:0] Threshold,
  input wire unsigned [15:0] MinPulseCount,
  output wire signed [15:0] DataOutA,
  output wire signed [15:0] DataOutB
);
// Registers with constant values to drive output to High or Low level
  reg signed [15:0] HI_LVL = 16'h7fff;
  reg signed [15:0] LO_LVL = 16'h0000;

// State Machine Definition
  parameter WaitForEdge = 2'b00;
  parameter TimeEvent = 2'b01;
  parameter EventEnded = 2'b11;
  parameter PeriodEnded = 2'b10;
  reg [1:0] State, NextState;

// Counter registers
  reg unsigned [31:0] PeriodCounter;
  reg unsigned [15:0] PulseCounter;
  reg unsigned [15:0] PulseLenCounter;

// Flags for trigger detection
  reg Triggered, Prev_Triggered, pulse_detected, QuantumState0;
  wire Trigger_edge;
  
  always @(posedge Clk) begin
    if (Reset==1'b1) begin
      PeriodCounter<=32'd0;
      PulseLenCounter<=16'd0;
      PulseCounter<=16'd0;
    end else begin
      if (PeriodCounter==PeriodCounterLimit)
        PeriodCounter<=32'd0;
      else
        PeriodCounter<=PeriodCounter+32'd1;

      if (State==TimeEvent)
        PulseLenCounter<=PulseLenCounter+16'd1;
      else
        PulseLenCounter<=16'd0;

      if (State==PeriodEnded)
        PulseCounter<=16'd0;
      else if (State==EventEnded & pulse_detected==1'b1)
        PulseCounter<=PulseCounter+16'd1;        
    end    
  end

  // Trigger: detect both threshold and edge trigger
  always @(posedge Clk) begin
    if (Reset==1'b1) begin
      Triggered<=1'b0;
      Prev_Triggered<=1'b0;
    end else begin
      Triggered<=(DataIn>Threshold)?1'b1:1'b0;
      Prev_Triggered<=Triggered;
    end
  end
  
  assign Trigger_edge = (~Prev_Triggered) & Triggered;

  // Move to next state
  always @(posedge Clk) begin
    if(Reset==1'b1)
      State <= WaitForEdge;
    else
      State <= NextState;
  end
  
  // Calculate next state
  always @(State or PeriodCounter or Triggered or PulseLenCounter or PulseCounter)
  begin
      case(State)
        WaitForEdge:begin
                      if (PeriodCounter==PeriodCounterLimit)
                        NextState<=PeriodEnded;
                      else if (Triggered==1'b1) 
                        NextState<=TimeEvent;
                      else
                        NextState<=WaitForEdge;
                      pulse_detected<=1'b0;
                    end  
        TimeEvent:begin
                      if (PeriodCounter==PeriodCounterLimit)
                        NextState<=PeriodEnded;
                      else if (Triggered ==1'b1)
                        NextState<=TimeEvent;
                      else
                        NextState<=EventEnded;
                      pulse_detected<=1'b0;
                  end
        EventEnded: begin
                      if (PeriodCounter==PeriodCounterLimit)
                        NextState<=PeriodEnded;
                      else if ((PulseLenCounter > PulseMin) & (PulseLenCounter < PulseMax)) begin
                        pulse_detected<=1'b1;
                        NextState<=WaitForEdge;
                      end
                      else begin
                        NextState<=WaitForEdge;
                        pulse_detected<=1'b0;
                      end
                    end 
        PeriodEnded:  begin
                        if (PulseCounter>=MinPulseCount)
                          QuantumState0<=1'b1;
                        else
                          QuantumState0<=1'b0;
                        pulse_detected<=1'b0;
                        NextState<=WaitForEdge;
                      end
        default: begin
        end
    endcase
  end

  assign DataOutA = (QuantumState0==1'b1)? HI_LVL:LO_LVL;
  assign DataOutB = (QuantumState0==1'b0)? HI_LVL:LO_LVL;

endmodule
