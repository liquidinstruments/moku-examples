library IEEE;
use IEEE.Std_Logic_1164.All;
use IEEE.Numeric_Std.all;

architecture DCSequencer of CustomWrapper is
begin
    DC_SEQUENCER: entity WORK.DCSequencer
        port map (
            Clk => Clk,
            Reset => Reset,
            DataIn => InputA,
            HIThreshold => signed(Control0(31 downto 16)),
            LOThreshold => signed(Control0(15 downto 0)),
            DataOutA => DataOutA,
            DataOutB => DataOutB
        );
end architecture;
