-----------------------------------------------------
--
-- Real Divide HDL Optimized
-- implimenting hardware module to perform fixed-point 
-- division of 'num' divided by 'den'
-- submodules used for validation, division, output reshaping
--
-----------------------------------------------------


LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;

ENTITY Real_Divide_HDL_Optimized IS
  PORT( clk                               :   IN    std_logic;
        reset                             :   IN    std_logic;
        num                               :   IN    std_logic_vector(15 DOWNTO 0);  -- int16
        den                               :   IN    std_logic_vector(15 DOWNTO 0);  -- int16
        validIn                           :   IN    std_logic;
        y                                 :   OUT   std_logic_vector(31 DOWNTO 0)  -- sfix32_En16
        );
END Real_Divide_HDL_Optimized;


ARCHITECTURE rtl OF Real_Divide_HDL_Optimized IS

  -- Component Declarations
  COMPONENT Verify_Divide_Sizes
    PORT( denominator                     :   IN    std_logic_vector(15 DOWNTO 0);  -- int16
          numerator                       :   IN    std_logic_vector(15 DOWNTO 0);  -- int16
          den                             :   OUT   std_logic_vector(15 DOWNTO 0);  -- int16
          num                             :   OUT   std_logic_vector(15 DOWNTO 0)  -- int16
          );
  END COMPONENT;

  COMPONENT For_Each_Real_Divide
    PORT( clk                             :   IN    std_logic;
          reset                           :   IN    std_logic;
          den                             :   IN    std_logic_vector(15 DOWNTO 0);  -- int16
          num                             :   IN    std_logic_vector(15 DOWNTO 0);  -- int16
          validIn                         :   IN    std_logic;
          y                               :   OUT   std_logic_vector(31 DOWNTO 0)  -- sfix32_En16
          );
  END COMPONENT;

  COMPONENT Reshape_to_Original_Size
    PORT( y                               :   IN    std_logic_vector(31 DOWNTO 0);  -- sfix32_En16
          u                               :   IN    std_logic_vector(15 DOWNTO 0);  -- int16
          z                               :   OUT   std_logic_vector(31 DOWNTO 0)  -- sfix32_En16
          );
  END COMPONENT;

  -- Component Configuration Statements
  FOR ALL : Verify_Divide_Sizes
    USE ENTITY work.Verify_Divide_Sizes(rtl);

  FOR ALL : For_Each_Real_Divide
    USE ENTITY work.For_Each_Real_Divide(rtl);

  FOR ALL : Reshape_to_Original_Size
    USE ENTITY work.Reshape_to_Original_Size(rtl);

  -- Signals
  SIGNAL den_1                            : std_logic_vector(15 DOWNTO 0);  -- ufix16
  SIGNAL num_1                            : std_logic_vector(15 DOWNTO 0);  -- ufix16
  SIGNAL ForEach_Real_Divide_out1         : std_logic_vector(31 DOWNTO 0);  -- ufix32
  SIGNAL z                                : std_logic_vector(31 DOWNTO 0);  -- ufix32
  SIGNAL z_signed                         : signed(31 DOWNTO 0);  -- sfix32_En16
  SIGNAL Delay17_out1                     : signed(31 DOWNTO 0) := to_signed(0, 32);  -- sfix32_En16

BEGIN
  u_Verify_divide_sizes : Verify_Divide_Sizes
    PORT MAP( denominator => den,  -- int16
              numerator => num,  -- int16
              den => den_1,  -- int16
              num => num_1  -- int16
              );

  u_ForEach_Real_Divide_instance1 : For_Each_Real_Divide
    PORT MAP( clk => clk,
              reset => reset,
              den => den_1,  -- int16
              num => num_1,  -- int16
              validIn => validIn,
              y => ForEach_Real_Divide_out1  -- sfix32_En16
              );

  u_Reshape_to_original_size : Reshape_to_Original_Size
    PORT MAP( y => ForEach_Real_Divide_out1,  -- sfix32_En16
              u => num,  -- int16
              z => z  -- sfix32_En16
              );

  z_signed <= signed(z);

  Delay17_process : PROCESS (clk)
  BEGIN
    IF clk'EVENT AND clk = '1' THEN
      Delay17_out1 <= z_signed;
    END IF;
  END PROCESS Delay17_process;


  y <= std_logic_vector(Delay17_out1);

END rtl;

