library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity Beamsplitter is
    Port (
        clk         : in  std_logic;
        rst         : in  std_logic;
        signal_in   : in  std_logic_vector(15 downto 0);
        noise_input : in  std_logic_vector(15 downto 0); -- 16-bit input
        out0        : out std_logic_vector(15 downto 0);
        out1        : out std_logic_vector(15 downto 0)
    );
end Beamsplitter;

architecture Behavioral of Beamsplitter is
    signal counter    : unsigned(3 downto 0) := (others => '0');
    signal temp0      : std_logic_vector(15 downto 0) := (others => '0');
    signal temp1      : std_logic_vector(15 downto 0) := (others => '0');
    signal noise_val  : std_logic; -- MSB of noise_input, used for positive/negative check
begin

    -- Extract the MSB of noise_input to determine if it's positive or negative
    noise_val <= noise_input(15);  -- MSB of noise_input

    -- Sample noise_input every 10 cycles
    process(clk, rst)
    begin
        if rst = '1' then
            counter <= (others => '0');
            temp0   <= (others => '0');
            temp1   <= (others => '0');
        elsif rising_edge(clk) then
            if counter = "1001" then  -- every 10 cycles
                -- If MSB of noise_input is '0' (positive), send signal to out0 (A), else to out1 (B)
                if noise_input(15) = '0' then
                    temp0 <= signal_in;  -- output A
                    temp1 <= (others => '0');
                else
                    temp0 <= (others => '0');
                    temp1 <= signal_in;  -- output B
                end if;

                -- Reset counter after 10 cycles
                counter <= (others => '0');
            else
                counter <= counter + 1;
            end if;
        end if;
    end process;

    -- Assign output signals
    out0 <= temp0;
    out1 <= temp1;

end Behavioral;
