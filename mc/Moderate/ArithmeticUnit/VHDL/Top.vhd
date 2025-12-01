library IEEE;
use IEEE.Std_Logic_1164.All;
use IEEE.Numeric_Std.all;

architecture AUWrapper of CustomWrapper is
begin
    AU_ENTITY: entity WORK.ArithmeticUnit
        port map (
            Clk => Clk,
            A => InputA,
            B => InputB,
            OpCode => Control1(1 downto 0),
            Result => OutputA
        );
end architecture;
