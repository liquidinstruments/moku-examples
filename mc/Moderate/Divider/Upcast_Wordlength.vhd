-----------------------------------------------------
--
-- Upcast Wordlength
-- converts input signals 'x' and 'y' to signed values
-- then converted back to std_logic_vector format and
-- and assigned to output forts 'u' and 'v' respectively
--
-----------------------------------------------------


LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;

ENTITY Upcast_Wordlength IS
  PORT( x                                 :   IN    std_logic_vector(15 DOWNTO 0);  -- int16
        y                                 :   IN    std_logic_vector(15 DOWNTO 0);  -- int16
        u                                 :   OUT   std_logic_vector(15 DOWNTO 0);  -- int16
        v                                 :   OUT   std_logic_vector(15 DOWNTO 0)  -- int16
        );
END Upcast_Wordlength;


ARCHITECTURE rtl OF Upcast_Wordlength IS

  -- Signals
  SIGNAL x_signed                         : signed(15 DOWNTO 0);  -- int16
  SIGNAL y_signed                         : signed(15 DOWNTO 0);  -- int16
  SIGNAL u_tmp                            : signed(15 DOWNTO 0);  -- int16
  SIGNAL v_tmp                            : signed(15 DOWNTO 0);  -- int16

BEGIN
  x_signed <= signed(x);

  y_signed <= signed(y);

  -- Copyright 2019 The MathWorks, Inc.
  u_tmp <= x_signed;
  v_tmp <= y_signed;

  u <= std_logic_vector(u_tmp);

  v <= std_logic_vector(v_tmp);

END rtl;

