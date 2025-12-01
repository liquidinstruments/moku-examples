library IEEE;
use IEEE.Std_Logic_1164.all;
use IEEE.Numeric_Std.all;

architecture Behavioural of CustomWrapper is
begin
    BOX: entity WORK.BoxcarAverager
        port map (
            Clk => Clk,
            Reset => Reset,
            InputA => InputA,
            InputB => InputB,
            OutputA => OutputA,
            OutputB => OutputB
        );
end architecture;
