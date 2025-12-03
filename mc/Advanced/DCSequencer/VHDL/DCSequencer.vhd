-----------------------------------------------------
--
-- Implementation of a DC sequencer
-- Sequence of DC levels are generated on DataOutA,
-- new level each time a trigger event occurs on DataIn.
-- Trigger has a Schmitt style trigger with a lo and hi
-- threshold. The cleaned trigger signal is output on
-- DataOutB for use downstream.
-- Configurable parameters:
--  Schmitt trigger high value (bits) = Control(0)(31 downto 16)
--  Schmitt trigger low value (bits) = Control(0)(15 downto 0)
--  Trigger output high and low values, constants
--  DC value sequence, array of constants
------------------------------------------------------


library IEEE;
use IEEE.Std_Logic_1164.all;
use IEEE.Numeric_Std.all;

entity DCSequencer is
    port (
        Clk : in std_logic;
        Reset : in std_logic;

        DataIn : in signed(15 downto 0);
        HIThreshold : in signed(15 downto 0);
        LOThreshold : in signed(15 downto 0);

        DataOutA : out signed(15 downto 0);
        DataOutB : out signed(15 downto 0)
    );
end entity;

architecture Behavioural of DCSequencer is
    type DCLevelsArray is array (0 to 127) of signed(15 downto 0);
    constant DC : DCLevelsArray := (
        x"371E", x"F110", x"F607", x"53A3", x"120E", x"F9EE", x"F2AD", x"F968",
        x"F6D1", x"D494", x"F5B8", x"0F78", x"0D18", x"DB9B", x"EDAA", x"E75A",
        x"EFF4", x"3BC6", x"E4EE", x"F99A", x"072A", x"F5D8", x"11CB", x"268F",
        x"27C1", x"E3C9", x"1F00", x"ED97", x"EBD7", x"20BA", x"DB56", x"F647",
        x"EA9B", x"F238", x"191F", x"D38C", x"DD0B", x"DC62", x"DD81", x"CBB7",
        x"3777", x"DE71", x"0500", x"2AB7", x"0F40", x"BB1D", x"EAF6", x"FFAE",
        x"F9E2", x"21B3", x"EEB1", x"AFD1", x"F48A", x"1069", x"1776", x"1315",
        x"CEE0", x"9D46", x"DEC7", x"DA28", x"05D2", x"4558", x"DFBA", x"4568",
        x"E9B1", x"F616", x"E688", x"182C", x"20DF", x"F6D8", x"09F8", x"D34D",
        x"24AD", x"4423", x"A9D3", x"E97F", x"2F27", x"0FA7", x"C64B", x"D25F",
        x"5DA9", x"F286", x"2591", x"F354", x"0CC9", x"B523", x"E7CC", x"106C",
        x"F2A3", x"16C7", x"24BD", x"EBB2", x"C42E", x"2008", x"FE75", x"F92C",
        x"D44F", x"24AC", x"0B6A", x"EC3E", x"F26E", x"F0E8", x"EB86", x"EEC3",
        x"25D1", x"F824", x"F31C", x"007B", x"E1B2", x"1AD9", x"285F", x"E38F",
        x"C6C3", x"0E85", x"2616", x"20B5", x"44AD", x"45F7", x"02FD", x"C927",
        x"1E7B", x"B58E", x"2270", x"23B5", x"C042", x"933F", x"044A", x"1E82"
    );

    constant HI_LVL : signed(15 downto 0) := x"7FFF";
    constant LO_LVL : signed(15 downto 0) := x"0000";

    signal Step : std_logic;
    signal Trigger, TriggerDly : std_logic;
    signal DCLevelAddr : unsigned(6 downto 0);
begin
    -- Input Schmitt trigger functionality
    -- to help reduce trigger on noise
    SCHMITT: process(Clk) is
    begin
        if rising_edge(Clk) then
            if Reset = '1' then
                Trigger <= '0';
            elsif DataIn >= HIThreshold then
                Trigger <= '1';
            elsif DataIn < LOThreshold then
                Trigger <= '0';
            end if;

            TriggerDly <= Trigger;
        end if;
    end process;

    -- Step the DC index on each rising edge
    Step <= Trigger and not TriggerDly;

    ADDR_COUNTER: process(Clk) is
    begin
        if rising_edge(Clk) then
            if Reset = '1' then
                DCLevelAddr <= (others => '0');
            elsif Step = '1' then
                -- Will wrap after 127
                DCLevelAddr <= DCLevelAddr + 1;
            end if;
        end if;
    end process;

    process(Clk) is
    begin
        if rising_edge(Clk) then
            DataOutA <= DC(to_integer(DCLevelAddr));
        end if;
    end process;

    DataOutB <= HI_LVL when Trigger = '1' else LO_LVL;
end architecture;
