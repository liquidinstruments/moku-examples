library IEEE;
use IEEE.Numeric_Std.all;

library Moku;
use Moku.Support.ScaleOffset;

-- Instantiate a DSP block using the ScaleOffset wrapper
architecture Behavioural of CustomInstrument is
begin
    -- Z = X * Scale + Offset
    -- Offset is units of bits, scale by default runs from -1 to 1 across whatever signal width is given
    -- Clips Z to min/max (prevents over/underflow)
    -- Includes rounding
    -- One Clock Cycle Delay
    DSP: ScaleOffset
        port map (
            Clk => Clk,
            Reset => Reset,
            X => InputA,
            Scale => signed(Control(0)(15 downto 0)),
            Offset => signed(Control(1)(15 downto 0)),
            Z => OutputA,
            Valid => '1',
            OutValid => open
        );

    -- If you want to change the range of the scale (e.g. multiply by more than 1), then set the
    -- NORMAL_SHIFT generic. This increases the range of Scale by 2^N, so NORMAL_SHIFT=4 means that
    -- the 16 bit scale here now covers the range -16 to 16.
    DSP_RANGE: ScaleOffset
        generic map (
            NORMAL_SHIFT => 4
        )
        port map (
            Clk => Clk,
            Reset => Reset,
            X => InputB,
            Scale => signed(Control(2)(15 downto 0)),
            Offset => signed(Control(3)(15 downto 0)),
            Z => OutputB,
            Valid => '1',
            OutValid => open
        );
end architecture;
