-- -------------------------------------------------------------
-- 
-- File Name: hdlsrc\System_Template\DSP.vhd
-- Created: 2021-09-30 12:46:01
-- 
-- Generated by MATLAB 9.11 and HDL Coder 3.19
-- 
-- 
-- -------------------------------------------------------------
-- Rate and Clocking Details
-- -------------------------------------------------------------
-- Model base rate: 3.2e-09
-- Target subsystem base rate: 3.2e-09
-- 
-- 
-- Clock Enable  Sample Time
-- -------------------------------------------------------------
-- ce_out_0      0
-- ce_out_1      3.2e-09
-- -------------------------------------------------------------
-- 
-- 
-- Output Signal                 Clock Enable  Sample Time
-- -------------------------------------------------------------
-- OutputA                       ce_out_0      0
-- OutputB                       ce_out_1      3.2e-09
-- -------------------------------------------------------------
-- 
-- -------------------------------------------------------------


-- -------------------------------------------------------------
-- 
-- Module: DSP
-- Source Path: System_Template/DSP
-- Hierarchy Level: 0
-- 
-- -------------------------------------------------------------
LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;

ENTITY DSP IS
  PORT( Clk                               :   IN    std_logic;
        Reset                             :   IN    std_logic;
        clk_enable                        :   IN    std_logic;
        InputA                            :   IN    signed(15 DOWNTO 0);  -- int16
        InputB                            :   IN    signed(15 DOWNTO 0);  -- int16
        ce_out_0                          :   OUT   std_logic;
        ce_out_1                          :   OUT   std_logic;
        OutputA                           :   OUT   signed(15 DOWNTO 0);  -- int16
        OutputB                           :   OUT   signed(15 DOWNTO 0)  -- int16
        );
END DSP;


ARCHITECTURE rtl OF DSP IS

  -- Component Declarations
  COMPONENT MATLAB_Function
    PORT( Clk                             :   IN    std_logic;
          Reset                           :   IN    std_logic;
          InputA                          :   IN    signed(15 DOWNTO 0);  -- int16
          OutputA                         :   OUT   unsigned(15 DOWNTO 0)  -- uint16
          );
  END COMPONENT;

  -- Component Configuration Statements
  FOR ALL : MATLAB_Function
    USE ENTITY work.MATLAB_Function(rtl);

  -- Signals
  SIGNAL MATLAB_Function_out1             : unsigned(15 DOWNTO 0);  -- uint16
  SIGNAL Data_Type_Conversion1_out1       : signed(15 DOWNTO 0);  -- int16
  SIGNAL switch_compare_1                 : std_logic;
  SIGNAL switch_compare_1_1               : std_logic;
  SIGNAL Relay_out1                       : signed(15 DOWNTO 0);  -- int16
  SIGNAL Relay_FB_sig                     : signed(15 DOWNTO 0);  -- int16
  SIGNAL Relay_conn_sig                   : signed(15 DOWNTO 0);  -- int16

BEGIN
  u_MATLAB_Function : MATLAB_Function
    PORT MAP( Clk => Clk,
              Reset => Reset,
              InputA => InputA,  -- int16
              OutputA => MATLAB_Function_out1  -- uint16
              );

  Data_Type_Conversion1_out1 <= signed(MATLAB_Function_out1);

  
  switch_compare_1 <= '1' WHEN InputB > to_signed(-16#0CCC#, 16) ELSE
      '0';

  
  switch_compare_1_1 <= '1' WHEN InputB >= to_signed(16#0CCC#, 16) ELSE
      '0';

  intdelay_process : PROCESS (Clk)
  BEGIN
    IF Clk'EVENT AND Clk = '1' THEN
      IF Reset = '1' THEN
        Relay_FB_sig <= to_signed(16#0000#, 16);
      ELSE 
        Relay_FB_sig <= Relay_out1;
      END IF;
    END IF;
  END PROCESS intdelay_process;


  
  Relay_conn_sig <= Relay_FB_sig WHEN switch_compare_1_1 = '0' ELSE
      to_signed(16#7FFF#, 16);

  
  Relay_out1 <= to_signed(16#0000#, 16) WHEN switch_compare_1 = '0' ELSE
      Relay_conn_sig;

  ce_out_0 <= clk_enable;

  ce_out_1 <= clk_enable;

  OutputA <= Data_Type_Conversion1_out1;

  OutputB <= Relay_out1;

END rtl;

