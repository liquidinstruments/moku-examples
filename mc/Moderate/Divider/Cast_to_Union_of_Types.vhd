-----------------------------------------------------
-- 
-- Cast to Union of Types
-- casts the second input to output as a union of the two input types
-- Cin is extended from a 16-bit number to a signed 32-bit 
-- integer by adding two additional zero bits 
-- Xref is converted to a signed format
-- 
-----------------------------------------------------


LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;

ENTITY Cast_to_Union_of_Types IS
  PORT( Xref                              :   IN    std_logic_vector(31 DOWNTO 0);  -- sfix32_En16
        Cin                               :   IN    std_logic_vector(15 DOWNTO 0);  -- sfix16_En14
        Cout                              :   OUT   std_logic_vector(31 DOWNTO 0)  -- sfix32_En16
        );
END Cast_to_Union_of_Types;


ARCHITECTURE rtl OF Cast_to_Union_of_Types IS

  -- Signals
  SIGNAL Xref_signed                      : signed(31 DOWNTO 0);  -- sfix32_En16
  SIGNAL Cin_signed                       : signed(15 DOWNTO 0);  -- sfix16_En14
  SIGNAL Cout_tmp                         : signed(31 DOWNTO 0);  -- sfix32_En16

BEGIN
  Xref_signed <= signed(Xref);

  Cin_signed <= signed(Cin);

  --castToUnionType Cast second input to union of the two types
  Cout_tmp <= resize(Cin_signed & '0' & '0', 32);

  Cout <= std_logic_vector(Cout_tmp);

END rtl;

