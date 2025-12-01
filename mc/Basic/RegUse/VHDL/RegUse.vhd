library IEEE;
use IEEE.Numeric_Std.all;

architecture Behavioural of CustomWrapper is
begin
    OutputA <= signed(Control1(15 downto 0));
    OutputB <= signed(Control2(15 downto 0));
end architecture;
