-----------------------------------------------------
--
-- Verify Divide Sizes
-- verifies the sizes of the input 'numerator' and 'denominator'
-- vectors, reshaping to 16-bit integer if required,
-- and outputting as a std_logic_vector as 'den' and 'num'
--
-----------------------------------------------------


LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;

ENTITY Verify_Divide_Sizes IS
  PORT( denominator                       :   IN    std_logic_vector(15 DOWNTO 0);  -- int16
        numerator                         :   IN    std_logic_vector(15 DOWNTO 0);  -- int16
        den                               :   OUT   std_logic_vector(15 DOWNTO 0);  -- int16
        num                               :   OUT   std_logic_vector(15 DOWNTO 0)  -- int16
        );
END Verify_Divide_Sizes;


ARCHITECTURE rtl OF Verify_Divide_Sizes IS

  -- Signals
  SIGNAL denominator_signed               : signed(15 DOWNTO 0);  -- int16
  SIGNAL numerator_signed                 : signed(15 DOWNTO 0);  -- int16
  SIGNAL den_tmp                          : signed(15 DOWNTO 0);  -- int16
  SIGNAL num_tmp                          : signed(15 DOWNTO 0);  -- int16

BEGIN
  denominator_signed <= signed(denominator);

  numerator_signed <= signed(numerator);

  -- Verify that the dimensions of the input arrays match, and then reshape
  -- them.

  den_tmp <= denominator_signed;
  num_tmp <= numerator_signed;

  den <= std_logic_vector(den_tmp);

  num <= std_logic_vector(num_tmp);

END rtl;

