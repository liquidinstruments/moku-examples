-----------------------------------------------------
--
-- MATLAB Function
-- right shifting the input signal 'num' by 1-bit sent 
-- to output 'num_out'. Input 't_num' has 1 subtracted from
-- it and is resized to 33-bits, output as 't_num_out'
-- used for fixed-point arithmetic in preparation for division
--
-----------------------------------------------------


LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;

ENTITY MATLAB_Function IS
  PORT( num                               :   IN    std_logic_vector(15 DOWNTO 0);  -- sfix16_En14
        t_num                             :   IN    std_logic_vector(31 DOWNTO 0);  -- int32
        num_out                           :   OUT   std_logic_vector(15 DOWNTO 0);  -- sfix16_En14
        t_num_out                         :   OUT   std_logic_vector(32 DOWNTO 0)  -- sfix33
        );
END MATLAB_Function;


ARCHITECTURE rtl OF MATLAB_Function IS

  -- Signals
  SIGNAL num_signed                       : signed(15 DOWNTO 0);  -- sfix16_En14
  SIGNAL t_num_signed                     : signed(31 DOWNTO 0);  -- int32
  SIGNAL num_out_tmp                      : signed(15 DOWNTO 0);  -- sfix16_En14
  SIGNAL t_num_out_tmp                    : signed(32 DOWNTO 0);  -- sfix33

BEGIN
  num_signed <= signed(num);

  t_num_signed <= signed(t_num);

  num_out_tmp <= SHIFT_RIGHT(num_signed, 1);
  t_num_out_tmp <= resize(t_num_signed, 33) - to_signed(1, 33);

  num_out <= std_logic_vector(num_out_tmp);

  t_num_out <= std_logic_vector(t_num_out_tmp);

END rtl;

