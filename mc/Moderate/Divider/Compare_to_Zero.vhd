-----------------------------------------------------
--
-- Compare to Zero
-- each real value is compared to zero
-- input is stored in variable 'u' and is compared to zero
-- result is stored in the ouput 'y'
-- 'y' flag is set to 1 when the input is greater than 0
--
-----------------------------------------------------


LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;

ENTITY Compare_To_Zero IS
  PORT( u                                 :   IN    std_logic_vector(15 DOWNTO 0);  -- sfix16_En14
        y                                 :   OUT   std_logic
        );
END Compare_To_Zero;


ARCHITECTURE rtl OF Compare_To_Zero IS

  -- Signals
  SIGNAL u_signed                         : signed(15 DOWNTO 0);  -- sfix16_En14
  SIGNAL Constant_out1                    : signed(15 DOWNTO 0);  -- sfix16_En14
  SIGNAL Compare_relop1                   : std_logic;

BEGIN
  u_signed <= signed(u);

  Constant_out1 <= to_signed(16#0000#, 16);

  
  Compare_relop1 <= '1' WHEN u_signed >= Constant_out1 ELSE
      '0';

  y <= Compare_relop1;

END rtl;

