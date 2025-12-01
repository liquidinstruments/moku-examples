library IEEE;
use IEEE.Numeric_Std.all;

library Moku;
use Moku.Support.ScaleOffset;

-- Instatiate a DSP block using the ScaleOffset wrapper
architecture Behavioural of CustomWrapper is
begin
    -- Z = X * Scale + Offset
    -- Offset is units of bits, scale by default runs from -1 to 1 across whatever signal width is given
    -- Clips Z to min/max (prevents over/underflow)
    -- Includes rounding
    -- One Clock Cycle Delay
    DSP1: ScaleOffset
        port map (
            Clk => Clk,
            Reset => Reset,
            X => InputA,
            Scale => signed(Control0(15 downto 0)),
            Offset => signed(Control1(15 downto 0)),
            Z => OutputA,
            Valid => '1',
            OutValid => open
        );

    -- If you want to change the range of the scale (e.g. multiply by more than 1), then set the
    -- NORMAL_SHIFT generic. This increases the range of Scale by 2^N, so NORMAL_SHIFT=4 means that
    -- the 16 bit scale here now covers the range -16 to 16.
    DSP2: ScaleOffset
        generic map (
            NORMAL_SHIFT => 4
        )
        port map (
            Clk => Clk,
            Reset => Reset,
            X => InputB,
            Scale => signed(Control2(15 downto 0)),
            Offset => signed(Control3(15 downto 0)),
            Z => OutputB,
            Valid => '1',
            OutValid => open
        );
end architecture;
