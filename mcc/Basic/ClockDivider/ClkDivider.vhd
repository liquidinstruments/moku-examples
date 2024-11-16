library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

-- Designed by Brian J. Neff / Liquid Instruments
-- This component will produce a clock divider 
-- The divider signal below can be adjusted to specify how many times you wish to divide the clock

entity ClkDivider is
    Port (
        clk         : in  STD_LOGIC;            -- Clock input
        reset       : in  STD_LOGIC;            -- Asynchronous reset
        pulse       : out STD_LOGIC             -- Timer pulse output
    );
end ClkDivider;

architecture Behavioral of ClkDivider is
    signal count   : INTEGER := 0;      -- Counter
    signal pulse_internal : STD_LOGIC;  -- Internal pulse signal
    signal divider : INTEGER := 1;      -- Sets how many times you would like to divide the clock in half.  All values less than one will divide clock in half, as you cannot logically divide it less than in half.
    signal max_count : INTEGER := 2;    -- Sets default as 2 but will be reset below.
begin
    process(reset, clk)
    begin
        if reset = '1' then
            count <= 0;
            pulse <= '0';

        elsif rising_edge(clk) then
            max_count <= divider - 1;
            if count >= max_count then
                pulse <= not pulse;
                count <= 0;
            else
                count <= count + 1;
            end if;
        end if;
    end process;

end Behavioral;