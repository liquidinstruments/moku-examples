library IEEE;
use IEEE.Numeric_Std.all;
library Moku;
use Moku.Support.clip_val;
use Moku.Support.sum_no_overflow;

architecture Behavioural of CustomWrapper is
    signal ch1_lower : signed(15 downto 0);
    signal ch1_upper : signed(15 downto 0);
    signal ch2_lower : signed(15 downto 0);
    signal ch2_upper : signed(15 downto 0);

begin
    ch1_lower <= signed(Control0(15 downto 0));
    ch1_upper <= signed(Control1(15 downto 0));
    ch2_lower <= signed(Control2(15 downto 0));
    ch2_upper <= signed(Control3(15 downto 0));

    -- Use library function to "clip" the value to within the upper and lower bounds
    OutputA <= clip_val(InputA, to_integer(ch1_lower), to_integer(ch1_upper));
    OutputB <= clip_val(InputB, to_integer(ch2_lower), to_integer(ch2_upper));
end architecture;
