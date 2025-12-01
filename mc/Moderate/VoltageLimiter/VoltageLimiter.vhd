library IEEE;
use IEEE.Numeric_Std.all;
library Moku;
use Moku.Support.clip_val;
use Moku.Support.sum_no_overflow;

architecture Behavioural of CustomWrapper is
    signal Ch1Lower : signed(16 downto 0);
    signal Ch1Upper : signed(16 downto 0);
    signal Ch2Lower : signed(16 downto 0);
    signal Ch2Upper : signed(16 downto 0);

begin
    Ch1Lower <= signed(Control0(16 downto 0));
    Ch1Upper <= signed(Control1(16 downto 0));
    Ch2Lower <= signed(Control2(16 downto 0));
    Ch2Upper <= signed(Control3(16 downto 0));

    -- Use library function to "clip" the value to within the upper and lower bounds
    OutputA <= clip_val(InputA, to_integer(Ch1Lower), to_integer(Ch1Upper));
    OutputB <= clip_val(InputB, to_integer(Ch2Lower), to_integer(Ch2Upper));
end architecture;
