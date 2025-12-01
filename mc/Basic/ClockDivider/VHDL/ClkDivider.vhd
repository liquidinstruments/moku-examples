library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

-- Designed by Brian J. Neff / Liquid Instruments
-- This component will produce a clock divider 
-- The divider signal below can be adjusted to specify how many times you wish to divide the clock

entity ClkDivider is
    generic(
        divider : unsigned(15 downto 0) := x"0001" --divide will default to x"0001" if no value is passed
    );
    Port (
        clk         : in  STD_LOGIC;            -- Clock input
        reset       : in  STD_LOGIC;            -- Asynchronous reset
        pulse       : out STD_LOGIC             -- Timer pulse output
    );
end ClkDivider;

architecture Behavioral of ClkDivider is
    signal count : unsigned(15 downto 0) := ( others => '0');
    signal pulse_internal : STD_LOGIC;  -- Internal pulse signal
    signal max_count : unsigned(15 downto 0);    
begin
    process(reset, clk)
    begin
        if reset = '1' then
            count <= ( others => '0');
            pulse <= '0';

        elsif rising_edge(clk) then
            max_count <= divider - 1;
            if count >= max_count then
                pulse <= not pulse;
                count <= ( others => '0');
            else
                count <= count + 1;
            end if;
        end if;
    end process;

end Behavioral;
