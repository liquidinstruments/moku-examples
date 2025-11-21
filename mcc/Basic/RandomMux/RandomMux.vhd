library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

architecture Behavioral of CustomWrapper is
    -- Beamsplitter internal signals
    signal counter     : unsigned(31 downto 0);
    signal cycle_limit : unsigned(31 downto 0);

begin
    cycle_limit <= unsigned(Control0); -- value of Control 0 detrmines length in cycles 
    -- Beamsplitter logic (inline)
    process(clk, Reset)
    begin
        if Reset = '1' then
            counter <= (others => '0');
            OutputA <= (others => '0');
            OutputB <= (others => '0');

        elsif rising_edge(clk) then
            if counter = cycle_limit then  -- every N cycles, from Control0
                if InputB(15) = '0' then    -- MSB = sign check
                    OutputA <= InputA;        -- send to output A
                    OutputB <= (others => '0');
                else
                    OutputA <= (others => '0');
                    OutputB <= InputA;        -- send to output B
                end if;
                counter <= (others => '0');

            else
                counter <= counter + 1;
            end if;
        end if;
    end process;

end Behavioral;
