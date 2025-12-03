-----------------------------------------------------
--
-- Normalize Numerator
-- normalizes the input integer 'u' into its fixed-point
-- representation 'x'
-- 'e' holds the scaling factor used in the normalization process
-- this flags if the input is negative and if the output is valid
--
-----------------------------------------------------


LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;

ENTITY Normalize_Numerator IS
  PORT( clk                               :   IN    std_logic;
        reset                             :   IN    std_logic;
        u                                 :   IN    std_logic_vector(15 DOWNTO 0);  -- int16
        validIn                           :   IN    std_logic;
        x                                 :   OUT   std_logic_vector(15 DOWNTO 0);  -- sfix16_En14
        e                                 :   OUT   std_logic_vector(31 DOWNTO 0);  -- int32
        isNegative                        :   OUT   std_logic
        );
END Normalize_Numerator;


ARCHITECTURE rtl OF Normalize_Numerator IS

  -- Component Declarations
  COMPONENT Positive_Real_Normalizer
    PORT( clk                             :   IN    std_logic;
          reset                           :   IN    std_logic;
          u                               :   IN    std_logic_vector(15 DOWNTO 0);  -- int16
          validIn                         :   IN    std_logic;
          x                               :   OUT   std_logic_vector(15 DOWNTO 0);  -- sfix16_En14
          e                               :   OUT   std_logic_vector(31 DOWNTO 0);  -- int32
          isNegative                      :   OUT   std_logic;
          validOut                        :   OUT   std_logic
          );
  END COMPONENT;

  -- Component Configuration Statements
  FOR ALL : Positive_Real_Normalizer
    USE ENTITY work.Positive_Real_Normalizer(rtl);

  -- Signals
  SIGNAL x_tmp                            : std_logic_vector(15 DOWNTO 0);  -- ufix16
  SIGNAL e_tmp                            : std_logic_vector(31 DOWNTO 0);  -- ufix32
  SIGNAL validOutdeadOut                  : std_logic;

BEGIN
  u_positiveRealNormalizer : Positive_Real_Normalizer
    PORT MAP( clk => clk,
              reset => reset,
              u => u,  -- int16
              validIn => validIn,
              x => x_tmp,  -- sfix16_En14
              e => e_tmp,  -- int32
              isNegative => isNegative,
              validOut => validOutdeadOut
              );

  x <= x_tmp;

  e <= e_tmp;

END rtl;

