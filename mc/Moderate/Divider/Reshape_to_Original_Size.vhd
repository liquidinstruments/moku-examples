-----------------------------------------------------
--
-- Reshape to Original Size
-- converts input signal 'y' a signed fixed-point, and 
-- 'u' to a signed integer representation. The conversion 
-- of 'y' is passed to the output port 'z'
--
-----------------------------------------------------


LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;

ENTITY Reshape_to_Original_Size IS
  PORT( y                                 :   IN    std_logic_vector(31 DOWNTO 0);  -- sfix32_En16
        u                                 :   IN    std_logic_vector(15 DOWNTO 0);  -- int16
        z                                 :   OUT   std_logic_vector(31 DOWNTO 0)  -- sfix32_En16
        );
END Reshape_to_Original_Size;


ARCHITECTURE rtl OF Reshape_to_Original_Size IS

  -- Signals
  SIGNAL y_signed                         : signed(31 DOWNTO 0);  -- sfix32_En16
  SIGNAL u_signed                         : signed(15 DOWNTO 0);  -- int16
  SIGNAL z_tmp                            : signed(31 DOWNTO 0);  -- sfix32_En16

BEGIN
  y_signed <= signed(y);

  u_signed <= signed(u);

  z_tmp <= y_signed;

  z <= std_logic_vector(z_tmp);

END rtl;

