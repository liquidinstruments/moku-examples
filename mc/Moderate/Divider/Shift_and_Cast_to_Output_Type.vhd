-----------------------------------------------------
--
-- Shift and Cast to Output Type
-- shifts the input 'x' either left or right, given the
-- value of 'shiftValue', then casts the result to the
-- output type
--
-----------------------------------------------------


LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;

ENTITY Shift_and_Cast_to_Output_Type IS
  PORT( clk                               :   IN    std_logic;
        x                                 :   IN    std_logic_vector(15 DOWNTO 0);  -- sfix16_En14
        shiftValue                        :   IN    std_logic_vector(33 DOWNTO 0);  -- sfix34
        y                                 :   OUT   std_logic_vector(31 DOWNTO 0)  -- sfix32_En16
        );
END Shift_and_Cast_to_Output_Type;


ARCHITECTURE rtl OF Shift_and_Cast_to_Output_Type IS

  -- Component Declarations
  COMPONENT Cast_to_Union_of_Types
    PORT( Xref                            :   IN    std_logic_vector(31 DOWNTO 0);  -- sfix32_En16
          Cin                             :   IN    std_logic_vector(15 DOWNTO 0);  -- sfix16_En14
          Cout                            :   OUT   std_logic_vector(31 DOWNTO 0)  -- sfix32_En16
          );
  END COMPONENT;

  COMPONENT Variable_Right_Shift
    PORT( clk                             :   IN    std_logic;
          x                               :   IN    std_logic_vector(31 DOWNTO 0);  -- sfix32_En16
          rightShift                      :   IN    std_logic_vector(31 DOWNTO 0);  -- int32
          y                               :   OUT   std_logic_vector(31 DOWNTO 0)  -- sfix32_En16
          );
  END COMPONENT;

  COMPONENT Variable_Left_Shift
    PORT( clk                             :   IN    std_logic;
          x                               :   IN    std_logic_vector(31 DOWNTO 0);  -- sfix32_En16
          leftShift                       :   IN    std_logic_vector(31 DOWNTO 0);  -- int32
          y                               :   OUT   std_logic_vector(31 DOWNTO 0)  -- sfix32_En16
          );
  END COMPONENT;

  -- Component Configuration Statements
  FOR ALL : Cast_to_Union_of_Types
    USE ENTITY work.Cast_to_Union_of_Types(rtl);

  FOR ALL : Variable_Right_Shift
    USE ENTITY work.Variable_Right_Shift(rtl);

  FOR ALL : Variable_Left_Shift
    USE ENTITY work.Variable_Left_Shift(rtl);

  -- Signals
  SIGNAL shiftValue_signed                : signed(33 DOWNTO 0);  -- sfix34
  SIGNAL Constant_out1                    : signed(31 DOWNTO 0);  -- sfix32_En16
  SIGNAL Cout                             : std_logic_vector(31 DOWNTO 0);  -- ufix32
  SIGNAL Abs_rsvd_y                       : signed(34 DOWNTO 0);  -- sfix35
  SIGNAL Abs_out1                         : signed(31 DOWNTO 0);  -- int32
  SIGNAL Delay3_out1                      : signed(33 DOWNTO 0) := to_signed(0, 34);  -- sfix34
  SIGNAL switch_compare_1                 : std_logic;
  SIGNAL Variable_Right_Shift_out1        : std_logic_vector(31 DOWNTO 0);  -- ufix32
  SIGNAL Variable_Right_Shift_out1_signed : signed(31 DOWNTO 0);  -- sfix32_En16
  SIGNAL Variable_Left_Shift_out1         : std_logic_vector(31 DOWNTO 0);  -- ufix32
  SIGNAL Variable_Left_Shift_out1_signed  : signed(31 DOWNTO 0);  -- sfix32_En16
  SIGNAL Switch_out1                      : signed(31 DOWNTO 0);  -- sfix32_En16
  SIGNAL Delay1_out1                      : signed(31 DOWNTO 0) := to_signed(0, 32);  -- sfix32_En16

BEGIN
  u_Cast_to_Union_of_Types : Cast_to_Union_of_Types
    PORT MAP( Xref => std_logic_vector(Constant_out1),  -- sfix32_En16
              Cin => x,  -- sfix16_En14
              Cout => Cout  -- sfix32_En16
              );

  u_Variable_Right_Shift : Variable_Right_Shift
    PORT MAP( clk => clk,
              x => Cout,  -- sfix32_En16
              rightShift => std_logic_vector(Abs_out1),  -- int32
              y => Variable_Right_Shift_out1  -- sfix32_En16
              );

  u_Variable_Left_Shift : Variable_Left_Shift
    PORT MAP( clk => clk,
              x => Cout,  -- sfix32_En16
              leftShift => std_logic_vector(Abs_out1),  -- int32
              y => Variable_Left_Shift_out1  -- sfix32_En16
              );

  shiftValue_signed <= signed(shiftValue);

  Constant_out1 <= to_signed(0, 32);

  
  Abs_rsvd_y <=  - (resize(shiftValue_signed, 35)) WHEN shiftValue_signed < to_signed(0, 34) ELSE
      resize(shiftValue_signed, 35);
  Abs_out1 <= Abs_rsvd_y(31 DOWNTO 0);

  Delay3_process : PROCESS (clk)
  BEGIN
    IF clk'EVENT AND clk = '1' THEN
      Delay3_out1 <= shiftValue_signed;
    END IF;
  END PROCESS Delay3_process;


  
  switch_compare_1 <= '1' WHEN Delay3_out1 >= to_signed(0, 34) ELSE
      '0';

  Variable_Right_Shift_out1_signed <= signed(Variable_Right_Shift_out1);

  Variable_Left_Shift_out1_signed <= signed(Variable_Left_Shift_out1);

  
  Switch_out1 <= Variable_Right_Shift_out1_signed WHEN switch_compare_1 = '0' ELSE
      Variable_Left_Shift_out1_signed;

  Delay1_process : PROCESS (clk)
  BEGIN
    IF clk'EVENT AND clk = '1' THEN
      Delay1_out1 <= Switch_out1;
    END IF;
  END PROCESS Delay1_process;


  y <= std_logic_vector(Delay1_out1);

END rtl;

