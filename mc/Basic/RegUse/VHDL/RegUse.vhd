library IEEE;
use IEEE.Numeric_Std.all;

architecture Behavioural of CustomInstrument is
begin
    OutputA <= signed(Control(1)(15 downto 0));
    OutputB <= signed(Control(2)(15 downto 0));
end architecture;
