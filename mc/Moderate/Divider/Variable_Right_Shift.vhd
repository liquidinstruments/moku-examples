-----------------------------------------------------
--
-- Variable Left Shift
-- performing left-shifting on the input signal 'x'
-- based on the value of 'leftShift', result being converted
-- to std_logic_vector format and stored in output port 'y'
--
-----------------------------------------------------


LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;

ENTITY Variable_Right_Shift IS
  PORT( clk                               :   IN    std_logic;
        x                                 :   IN    std_logic_vector(31 DOWNTO 0);  -- sfix32_En16
        rightShift                        :   IN    std_logic_vector(31 DOWNTO 0);  -- int32
        y                                 :   OUT   std_logic_vector(31 DOWNTO 0)  -- sfix32_En16
        );
END Variable_Right_Shift;


ARCHITECTURE rtl OF Variable_Right_Shift IS

  -- Signals
  SIGNAL rightShift_signed                : signed(31 DOWNTO 0);  -- int32
  SIGNAL Bit_Slice8_out1                  : std_logic;  -- ufix1
  SIGNAL Delay4_out1                      : std_logic := '0';  -- ufix1
  SIGNAL Bit_Slice7_out1                  : std_logic;  -- ufix1
  SIGNAL Delay3_out1                      : std_logic := '0';  -- ufix1
  SIGNAL Bit_Slice6_out1                  : std_logic;  -- ufix1
  SIGNAL Delay2_out1                      : std_logic := '0';  -- ufix1
  SIGNAL Bit_Slice5_out1                  : std_logic;  -- ufix1
  SIGNAL Delay1_out1                      : std_logic := '0';  -- ufix1
  SIGNAL Bit_Slice4_out1                  : std_logic;  -- ufix1
  SIGNAL Bit_Slice3_out1                  : std_logic;  -- ufix1
  SIGNAL Bit_Slice2_out1                  : std_logic;  -- ufix1
  SIGNAL Bit_Slice1_out1                  : std_logic;  -- ufix1
  SIGNAL x_signed                         : signed(31 DOWNTO 0);  -- sfix32_En16
  SIGNAL Bit_Shift_out1                   : signed(31 DOWNTO 0);  -- sfix32_En16
  SIGNAL Switch1_out1                     : signed(31 DOWNTO 0);  -- sfix32_En16
  SIGNAL Bit_Shift1_out1                  : signed(31 DOWNTO 0);  -- sfix32_En16
  SIGNAL Switch2_out1                     : signed(31 DOWNTO 0);  -- sfix32_En16
  SIGNAL Bit_Shift2_out1                  : signed(31 DOWNTO 0);  -- sfix32_En16
  SIGNAL Switch3_out1                     : signed(31 DOWNTO 0);  -- sfix32_En16
  SIGNAL Bit_Shift3_out1                  : signed(31 DOWNTO 0);  -- sfix32_En16
  SIGNAL Switch4_out1                     : signed(31 DOWNTO 0);  -- sfix32_En16
  SIGNAL Delay_out1                       : signed(31 DOWNTO 0) := to_signed(0, 32);  -- sfix32_En16
  SIGNAL Bit_Shift4_out1                  : signed(31 DOWNTO 0);  -- sfix32_En16
  SIGNAL Switch5_out1                     : signed(31 DOWNTO 0);  -- sfix32_En16
  SIGNAL Bit_Shift5_out1                  : signed(31 DOWNTO 0);  -- sfix32_En16
  SIGNAL Switch6_out1                     : signed(31 DOWNTO 0);  -- sfix32_En16
  SIGNAL Bit_Shift6_out1                  : signed(31 DOWNTO 0);  -- sfix32_En16
  SIGNAL Switch7_out1                     : signed(31 DOWNTO 0);  -- sfix32_En16
  SIGNAL Bit_Shift7_out1                  : signed(31 DOWNTO 0);  -- sfix32_En16
  SIGNAL Switch8_out1                     : signed(31 DOWNTO 0);  -- sfix32_En16

BEGIN
  rightShift_signed <= signed(rightShift);

  Bit_Slice8_out1 <= rightShift_signed(7);

  Delay4_process : PROCESS (clk)
  BEGIN
    IF clk'EVENT AND clk = '1' THEN
      Delay4_out1 <= Bit_Slice8_out1;
    END IF;
  END PROCESS Delay4_process;


  Bit_Slice7_out1 <= rightShift_signed(6);

  Delay3_process : PROCESS (clk)
  BEGIN
    IF clk'EVENT AND clk = '1' THEN
      Delay3_out1 <= Bit_Slice7_out1;
    END IF;
  END PROCESS Delay3_process;


  Bit_Slice6_out1 <= rightShift_signed(5);

  Delay2_process : PROCESS (clk)
  BEGIN
    IF clk'EVENT AND clk = '1' THEN
      Delay2_out1 <= Bit_Slice6_out1;
    END IF;
  END PROCESS Delay2_process;


  Bit_Slice5_out1 <= rightShift_signed(4);

  Delay1_process : PROCESS (clk)
  BEGIN
    IF clk'EVENT AND clk = '1' THEN
      Delay1_out1 <= Bit_Slice5_out1;
    END IF;
  END PROCESS Delay1_process;


  Bit_Slice4_out1 <= rightShift_signed(3);

  Bit_Slice3_out1 <= rightShift_signed(2);

  Bit_Slice2_out1 <= rightShift_signed(1);

  Bit_Slice1_out1 <= rightShift_signed(0);

  x_signed <= signed(x);

  Bit_Shift_out1 <= SHIFT_RIGHT(x_signed, 1);

  
  Switch1_out1 <= x_signed WHEN Bit_Slice1_out1 = '0' ELSE
      Bit_Shift_out1;

  Bit_Shift1_out1 <= SHIFT_RIGHT(Switch1_out1, 2);

  
  Switch2_out1 <= Switch1_out1 WHEN Bit_Slice2_out1 = '0' ELSE
      Bit_Shift1_out1;

  Bit_Shift2_out1 <= SHIFT_RIGHT(Switch2_out1, 4);

  
  Switch3_out1 <= Switch2_out1 WHEN Bit_Slice3_out1 = '0' ELSE
      Bit_Shift2_out1;

  Bit_Shift3_out1 <= SHIFT_RIGHT(Switch3_out1, 8);

  
  Switch4_out1 <= Switch3_out1 WHEN Bit_Slice4_out1 = '0' ELSE
      Bit_Shift3_out1;

  Delay_process : PROCESS (clk)
  BEGIN
    IF clk'EVENT AND clk = '1' THEN
      Delay_out1 <= Switch4_out1;
    END IF;
  END PROCESS Delay_process;


  Bit_Shift4_out1 <= SHIFT_RIGHT(Delay_out1, 16);

  
  Switch5_out1 <= Delay_out1 WHEN Delay1_out1 = '0' ELSE
      Bit_Shift4_out1;

  Bit_Shift5_out1 <= SHIFT_RIGHT(Switch5_out1, 31);

  
  Switch6_out1 <= Switch5_out1 WHEN Delay2_out1 = '0' ELSE
      Bit_Shift5_out1;

  Bit_Shift6_out1 <= SHIFT_RIGHT(Switch6_out1, 31);

  
  Switch7_out1 <= Switch6_out1 WHEN Delay3_out1 = '0' ELSE
      Bit_Shift6_out1;

  Bit_Shift7_out1 <= SHIFT_RIGHT(Switch7_out1, 31);

  
  Switch8_out1 <= Switch7_out1 WHEN Delay4_out1 = '0' ELSE
      Bit_Shift7_out1;

  y <= std_logic_vector(Switch8_out1);

END rtl;

