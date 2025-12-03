-----------------------------------------------------
--
-- Normalize Denominator
-- normalizes the input integer 'u' into its fixed-point
-- representation 'x'
-- 'e' holds the scaling factor used in the normalization process
-- this flags if the input is negative and if the output is valid
--
-----------------------------------------------------

LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;

ENTITY Normalize_Denominator IS
  PORT( clk                               :   IN    std_logic;
        reset                             :   IN    std_logic;
        u                                 :   IN    std_logic_vector(15 DOWNTO 0);  -- int16
        validIn                           :   IN    std_logic;
        x                                 :   OUT   std_logic_vector(15 DOWNTO 0);  -- sfix16_En14
        e                                 :   OUT   std_logic_vector(31 DOWNTO 0);  -- int32
        isNegative                        :   OUT   std_logic;
        validOut                          :   OUT   std_logic
        );
END Normalize_Denominator;


ARCHITECTURE rtl OF Normalize_Denominator IS

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
  SIGNAL positiveRealNormalizer_out1      : std_logic_vector(15 DOWNTO 0);  -- ufix16
  SIGNAL positiveRealNormalizer_out2      : std_logic_vector(31 DOWNTO 0);  -- ufix32
  SIGNAL positiveRealNormalizer_out3      : std_logic;
  SIGNAL positiveRealNormalizer_out4      : std_logic;

BEGIN
  u_positiveRealNormalizer : Positive_Real_Normalizer
    PORT MAP( clk => clk,
              reset => reset,
              u => u,  -- int16
              validIn => validIn,
              x => positiveRealNormalizer_out1,  -- sfix16_En14
              e => positiveRealNormalizer_out2,  -- int32
              isNegative => positiveRealNormalizer_out3,
              validOut => positiveRealNormalizer_out4
              );

  x <= positiveRealNormalizer_out1;

  e <= positiveRealNormalizer_out2;

  isNegative <= positiveRealNormalizer_out3;

  validOut <= positiveRealNormalizer_out4;

END rtl;

