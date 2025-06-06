-- -------------------------------------------------------------
-- 
-- File Name: hdl_prj\hdlsrc\BoxcarAveragerFixedPointPublish\Boxcar_FixPt.vhd
-- Created: 2024-10-21 19:53:33
-- 
-- Generated by MATLAB 23.2, HDL Coder 23.2, and Simulink 23.2
-- 
-- -------------------------------------------------------------


-- -------------------------------------------------------------
-- 
-- Module: Boxcar_FixPt
-- Source Path: BoxcarAveragerFixedPointPublish/DSP/Boxcar_FixPt
-- Hierarchy Level: 1
-- Model version: 6.10
-- 
-- -------------------------------------------------------------
LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;

ENTITY Boxcar_FixPt IS
  PORT( Clk                               :   IN    std_logic;
        Reset                             :   IN    std_logic;
        SignalInput                       :   IN    signed(15 DOWNTO 0);  -- int16
        Trigger                           :   IN    std_logic;
        TriggerDelay                      :   IN    unsigned(15 DOWNTO 0);  -- uint16
        GateLength                        :   IN    unsigned(15 DOWNTO 0);  -- uint16
        AverageOut                        :   OUT   signed(31 DOWNTO 0);  -- int32
        AveragingFlag                     :   OUT   std_logic;  -- ufix1
        DataValid                         :   OUT   std_logic  -- ufix1
        );
END Boxcar_FixPt;


ARCHITECTURE rtl OF Boxcar_FixPt IS

  -- Signals
  SIGNAL AvgOut                           : signed(31 DOWNTO 0);  -- sfix32
  SIGNAL cnt1                             : unsigned(15 DOWNTO 0);  -- ufix16
  SIGNAL cnt2                             : unsigned(15 DOWNTO 0);  -- ufix16
  SIGNAL AvgFlag                          : std_logic;  -- ufix1
  SIGNAL DValid                           : std_logic;  -- ufix1
  SIGNAL current_state                    : unsigned(1 DOWNTO 0);  -- ufix2
  SIGNAL AvgOut_next                      : signed(31 DOWNTO 0);  -- sfix32
  SIGNAL cnt1_next                        : unsigned(15 DOWNTO 0);  -- ufix16
  SIGNAL cnt2_next                        : unsigned(15 DOWNTO 0);  -- ufix16
  SIGNAL AvgFlag_next                     : std_logic;  -- ufix1
  SIGNAL DValid_next                      : std_logic;  -- ufix1
  SIGNAL current_state_next               : unsigned(1 DOWNTO 0);  -- ufix2

BEGIN
  Boxcar_FixPt_1_process : PROCESS (Clk)
  BEGIN
    IF Clk'EVENT AND Clk = '1' THEN
      IF Reset = '1' THEN
        AvgOut <= to_signed(0, 32);
        cnt1 <= to_unsigned(16#0000#, 16);
        cnt2 <= to_unsigned(16#0000#, 16);
        AvgFlag <= '0';
        DValid <= '0';
        current_state <= to_unsigned(16#0#, 2);
      ELSE 
        AvgOut <= AvgOut_next;
        cnt1 <= cnt1_next;
        cnt2 <= cnt2_next;
        AvgFlag <= AvgFlag_next;
        DValid <= DValid_next;
        current_state <= current_state_next;
      END IF;
    END IF;
  END PROCESS Boxcar_FixPt_1_process;

  Boxcar_FixPt_1_output : PROCESS (AvgFlag, AvgOut, DValid, GateLength, SignalInput, Trigger, TriggerDelay, cnt1,
       cnt2, current_state)
    VARIABLE GateLength1 : unsigned(15 DOWNTO 0);
    VARIABLE SignalInput1 : signed(15 DOWNTO 0);
    VARIABLE TriggerDelay1 : unsigned(15 DOWNTO 0);
    VARIABLE AvgOut_temp : signed(31 DOWNTO 0);
    VARIABLE cnt2_temp : unsigned(15 DOWNTO 0);
    VARIABLE AvgFlag_temp : std_logic;
    VARIABLE DValid_temp : std_logic;
  BEGIN
    AvgOut_temp := AvgOut;
    cnt2_temp := cnt2;
    AvgFlag_temp := AvgFlag;
    DValid_temp := DValid;
    cnt1_next <= cnt1;
    current_state_next <= current_state;
    --MATLAB Function 'DSP/Boxcar_FixPt'
    --auto-generated
    GateLength1 := GateLength;
    SignalInput1 := SignalInput;
    TriggerDelay1 := TriggerDelay;
    --Waiting for trigger
    CASE current_state IS
      WHEN "00" =>
        DValid_temp := '0';
        AvgFlag_temp := '0';
        IF Trigger = '1' THEN 
          current_state_next <= to_unsigned(16#1#, 2);
          cnt1_next <= TriggerDelay1;
        ELSE 
          current_state_next <= to_unsigned(16#0#, 2);
        END IF;
      WHEN "01" =>
        IF cnt1 = to_unsigned(16#0000#, 16) THEN 
          current_state_next <= to_unsigned(16#2#, 2);
          cnt2_temp := GateLength1;
          AvgOut_temp := to_signed(0, 32);
        ELSE 
          cnt1_next <= cnt1 - to_unsigned(16#0001#, 16);
        END IF;
      WHEN "10" =>
        IF cnt2 = to_unsigned(16#0000#, 16) THEN 
          current_state_next <= to_unsigned(16#0#, 2);
          DValid_temp := '0';
        ELSE 
          AvgOut_temp := AvgOut + resize(SignalInput1, 32);
          AvgFlag_temp := '1';
          cnt2_temp := cnt2 - to_unsigned(16#0001#, 16);
          IF cnt2_temp = to_unsigned(16#0000#, 16) THEN 
            DValid_temp := '1';
            current_state_next <= to_unsigned(16#0#, 2);
          END IF;
        END IF;
      WHEN OTHERS => 
        NULL;
    END CASE;
    AverageOut <= AvgOut_temp;
    AveragingFlag <= AvgFlag_temp;
    DataValid <= DValid_temp;
    AvgOut_next <= AvgOut_temp;
    cnt2_next <= cnt2_temp;
    AvgFlag_next <= AvgFlag_temp;
    DValid_next <= DValid_temp;
  END PROCESS Boxcar_FixPt_1_output;


END rtl;

