-----------------------------------------------------
--
-- DSP
-- OutputA = Control(0) * InputA / InputB
-- Control(0) must be set to 1 to enable the output and 
-- can be scaled in the CC Instrument to scale the result
--
-----------------------------------------------------

-----------------------------------------------------
-- Rate and Clocking Details
-----------------------------------------------------
-- Model base rate: 3.2e-09
-- Target subsystem base rate: 3.2e-09
-- 
-----------------------------------------------------

LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;

ENTITY DSP IS
  PORT( clk                               :   IN    std_logic;
        reset                             :   IN    std_logic;
        InputA                            :   IN    std_logic_vector(15 DOWNTO 0);  -- int16
        InputB                            :   IN    std_logic_vector(15 DOWNTO 0);  -- int16
        Control(0)                        :   IN    std_logic_vector(15 DOWNTO 0);  -- int16
        OutputA                           :   OUT   std_logic_vector(15 DOWNTO 0)  -- int16
        );
END DSP;


ARCHITECTURE rtl OF DSP IS

  -- Component Declarations
  COMPONENT Real_Divide_HDL_Optimized
    PORT( clk                             :   IN    std_logic;
          reset                           :   IN    std_logic;
          num                             :   IN    std_logic_vector(15 DOWNTO 0);  -- int16
          den                             :   IN    std_logic_vector(15 DOWNTO 0);  -- int16
          validIn                         :   IN    std_logic;
          y                               :   OUT   std_logic_vector(31 DOWNTO 0)  -- sfix32_En16
          );
  END COMPONENT;

  -- Component Configuration Statements
  FOR ALL : Real_Divide_HDL_Optimized
    USE ENTITY work.Real_Divide_HDL_Optimized(rtl);

  -- Signals
  SIGNAL Constant_out1                    : std_logic;
  SIGNAL Real_Divide_HDL_Optimized_out1   : std_logic_vector(31 DOWNTO 0);  -- ufix32
  SIGNAL Real_Divide_HDL_Optimized_out1_signed : signed(31 DOWNTO 0);  -- sfix32_En16
  SIGNAL Delay_out1                       : signed(31 DOWNTO 0) := to_signed(0, 32);  -- sfix32_En16
  SIGNAL Control0_signed                  : signed(15 DOWNTO 0);  -- int16
  SIGNAL Data_Type_Conversion_out1        : unsigned(15 DOWNTO 0);  -- uint16
  SIGNAL Delay1_out1                      : unsigned(15 DOWNTO 0) := to_unsigned(16#0000#, 16);  -- uint16
  SIGNAL Product_cast                     : signed(16 DOWNTO 0);  -- sfix17
  SIGNAL Product_mul_temp                 : signed(48 DOWNTO 0);  -- sfix49_En16
  SIGNAL Product_cast_1                   : signed(47 DOWNTO 0);  -- sfix48_En16
  SIGNAL Product_out1                     : signed(15 DOWNTO 0);  -- int16
  SIGNAL Delay2_out1                      : signed(15 DOWNTO 0) := to_signed(16#0000#, 16);  -- int16

BEGIN
  u_Real_Divide_HDL_Optimized : Real_Divide_HDL_Optimized
    PORT MAP( clk => clk,
              reset => reset,
              num => InputA,  -- int16
              den => InputB,  -- int16
              validIn => Constant_out1,
              y => Real_Divide_HDL_Optimized_out1  -- sfix32_En16
              );

  Constant_out1 <= '1';

  Real_Divide_HDL_Optimized_out1_signed <= signed(Real_Divide_HDL_Optimized_out1);

  Delay_process : PROCESS (clk)
  BEGIN
    IF clk'EVENT AND clk = '1' THEN
      Delay_out1 <= Real_Divide_HDL_Optimized_out1_signed;
    END IF;
  END PROCESS Delay_process;


  Control0_signed <= signed(Control(0));

  Data_Type_Conversion_out1 <= unsigned(Control0_signed);

  Delay1_process : PROCESS (clk)
  BEGIN
    IF clk'EVENT AND clk = '1' THEN
      Delay1_out1 <= Data_Type_Conversion_out1;
    END IF;
  END PROCESS Delay1_process;


  Product_cast <= signed(resize(Delay1_out1, 17));
  Product_mul_temp <= Delay_out1 * Product_cast;
  Product_cast_1 <= Product_mul_temp(47 DOWNTO 0);
  
  Product_out1 <= X"7FFF" WHEN (Product_cast_1(47) = '0') AND (Product_cast_1(46 DOWNTO 31) /= X"0000") ELSE
      X"8000" WHEN (Product_cast_1(47) = '1') AND (Product_cast_1(46 DOWNTO 31) /= X"FFFF") ELSE
      Product_cast_1(31 DOWNTO 16);

  Delay2_process : PROCESS (clk)
  BEGIN
    IF clk'EVENT AND clk = '1' THEN
      Delay2_out1 <= Product_out1;
    END IF;
  END PROCESS Delay2_process;


  OutputA <= std_logic_vector(Delay2_out1);

END rtl;

