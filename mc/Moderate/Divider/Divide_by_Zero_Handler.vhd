-----------------------------------------------------
-- 
-- Divide by Zero Handler
-- handles divide-by-zero scenarios
-- compares the input signal 'yIn' against zero and 
-- outputs a constant based on the result
-- if 'yIn' > 0 : outputs the max negative 32-bit integer
-- otherwise : outputs the max positive 32-bit integer
-- 
-----------------------------------------------------


LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;

ENTITY Divide_by_Zero_Handler IS
  PORT( clk                               :   IN    std_logic;
        yIn                               :   IN    std_logic_vector(15 DOWNTO 0);  -- sfix16_En14
        yOut                              :   OUT   std_logic_vector(31 DOWNTO 0)  -- sfix32_En16
        );
END Divide_by_Zero_Handler;


ARCHITECTURE rtl OF Divide_by_Zero_Handler IS

  -- Component Declarations
  COMPONENT Compare_To_Zero
    PORT( u                               :   IN    std_logic_vector(15 DOWNTO 0);  -- sfix16_En14
          y                               :   OUT   std_logic
          );
  END COMPONENT;

  -- Component Configuration Statements
  FOR ALL : Compare_To_Zero
    USE ENTITY work.Compare_To_Zero(rtl);

  -- Signals
  SIGNAL Compare_To_Zero_out1             : std_logic;
  SIGNAL Delay13_out1                     : std_logic := '0';
  SIGNAL switch_compare_1                 : std_logic;
  SIGNAL Constant3_out1                   : signed(31 DOWNTO 0);  -- sfix32_En16
  SIGNAL Constant4_out1                   : signed(31 DOWNTO 0);  -- sfix32_En16
  SIGNAL Switch_out1                      : signed(31 DOWNTO 0);  -- sfix32_En16
  SIGNAL Delay3_out1                      : signed(31 DOWNTO 0) := to_signed(0, 32);  -- sfix32_En16

BEGIN
  u_Compare_To_Zero : Compare_To_Zero
    PORT MAP( u => yIn,  -- sfix16_En14
              y => Compare_To_Zero_out1
              );

  Delay13_process : PROCESS (clk)
  BEGIN
    IF clk'EVENT AND clk = '1' THEN
      Delay13_out1 <= Compare_To_Zero_out1;
    END IF;
  END PROCESS Delay13_process;


  
  switch_compare_1 <= '1' WHEN Delay13_out1 > '0' ELSE
      '0';

  Constant3_out1 <= signed'(X"80000000");

  Constant4_out1 <= to_signed(2147483647, 32);

  
  Switch_out1 <= Constant3_out1 WHEN switch_compare_1 = '0' ELSE
      Constant4_out1;

  Delay3_process : PROCESS (clk)
  BEGIN
    IF clk'EVENT AND clk = '1' THEN
      Delay3_out1 <= Switch_out1;
    END IF;
  END PROCESS Delay3_process;


  yOut <= std_logic_vector(Delay3_out1);

END rtl;

