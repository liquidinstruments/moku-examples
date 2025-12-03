library IEEE;
use IEEE.Numeric_Std.all;

library Moku;
use Moku.Support.clip;

architecture Behavioural of CustomInstrument is 
begin
    OutputA <= resize(clip(InputA, 8, 0), 16); 
end architecture;
