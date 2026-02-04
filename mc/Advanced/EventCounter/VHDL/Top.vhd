library IEEE;
use IEEE.Std_Logic_1164.All;
use IEEE.Numeric_Std.all;

architecture EventCounter of CustomInstrument is
begin
    EVENT_COUNTER: entity WORK.EventCounter
        port map (
            Clk => Clk,
            Reset => Reset,

            DataIn => InputA,
            PeriodCounterLimit => unsigned(Control(0)),
            PulseMin => unsigned(Control(1)(15 downto 0)),
            PulseMax => unsigned(Control(1)(31 downto 16)),
            Threshold => signed(Control(2)(15 downto 0)),
            MinPulseCount => unsigned(Control(2)(31 downto 16)),

            DataOutA => OutputA,
            DataOutB => OutputB
        );
end architecture;
