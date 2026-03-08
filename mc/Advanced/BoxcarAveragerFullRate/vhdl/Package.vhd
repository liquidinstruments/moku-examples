library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;

package Types is
    type signed16_array is array (natural range <>) of signed(15 downto 0);
    type signed_sum_array is array (natural range <>) of signed;
    -- Note: Some VHDL compilers require a fixed width in the package, 
    -- if so, use: type signed48_array is array (natural range <>) of signed(47 downto 0);
    -- type t_ram is array (0 to (2**G_RAM_ADDR_WIDTH) - 1) of signed(C_IN_WIDTH - 1 downto 0);

end package Types;