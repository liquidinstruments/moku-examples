-----------------------------------------------------
--
-- Divide Real Numerator by Denominator 
-- wrapper for normalized_CORDIC_Divide
--
-----------------------------------------------------


LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;

ENTITY Divide_Real_Numerator_by_Denominator IS
  PORT( clk                               :   IN    std_logic;
        reset                             :   IN    std_logic;
        num                               :   IN    std_logic_vector(15 DOWNTO 0);  -- sfix16_En14
        tNum                              :   IN    std_logic_vector(32 DOWNTO 0);  -- sfix33
        den                               :   IN    std_logic_vector(15 DOWNTO 0);  -- sfix16_En14
        tDen                              :   IN    std_logic_vector(31 DOWNTO 0);  -- int32
        isNumNegative                     :   IN    std_logic;
        isDenNegative                     :   IN    std_logic;
        validIn                           :   IN    std_logic;
        y                                 :   OUT   std_logic_vector(15 DOWNTO 0);  -- sfix16_En14
        e                                 :   OUT   std_logic_vector(33 DOWNTO 0);  -- sfix34
        isDenZero                         :   OUT   std_logic;
        validOut                          :   OUT   std_logic
        );
END Divide_Real_Numerator_by_Denominator;


ARCHITECTURE rtl OF Divide_Real_Numerator_by_Denominator IS

  -- Component Declarations
  COMPONENT Normalized_CORDIC_Divide
    PORT( clk                             :   IN    std_logic;
          reset                           :   IN    std_logic;
          num                             :   IN    std_logic_vector(15 DOWNTO 0);  -- sfix16_En14
          den                             :   IN    std_logic_vector(15 DOWNTO 0);  -- sfix16_En14
          tNum                            :   IN    std_logic_vector(32 DOWNTO 0);  -- sfix33
          tDen                            :   IN    std_logic_vector(31 DOWNTO 0);  -- int32
          isNumNegative                   :   IN    std_logic;
          isDenNegative                   :   IN    std_logic;
          validIn                         :   IN    std_logic;
          y                               :   OUT   std_logic_vector(15 DOWNTO 0);  -- sfix16_En14
          t                               :   OUT   std_logic_vector(33 DOWNTO 0);  -- sfix34
          isDenZeroOut                    :   OUT   std_logic;
          validOut                        :   OUT   std_logic
          );
  END COMPONENT;

  -- Component Configuration Statements
  FOR ALL : Normalized_CORDIC_Divide
    USE ENTITY work.Normalized_CORDIC_Divide(rtl);

  -- Signals
  SIGNAL y_tmp                            : std_logic_vector(15 DOWNTO 0);  -- ufix16
  SIGNAL t                                : std_logic_vector(33 DOWNTO 0);  -- ufix34
  SIGNAL isDenZeroOut                     : std_logic;

BEGIN
  u_normalizedCORDICDivide : Normalized_CORDIC_Divide
    PORT MAP( clk => clk,
              reset => reset,
              num => num,  -- sfix16_En14
              den => den,  -- sfix16_En14
              tNum => tNum,  -- sfix33
              tDen => tDen,  -- int32
              isNumNegative => isNumNegative,
              isDenNegative => isDenNegative,
              validIn => validIn,
              y => y_tmp,  -- sfix16_En14
              t => t,  -- sfix34
              isDenZeroOut => isDenZeroOut,
              validOut => validOut
              );

  y <= y_tmp;

  e <= t;

  isDenZero <= isDenZeroOut;

END rtl;

