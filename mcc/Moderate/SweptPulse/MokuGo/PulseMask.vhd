library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity PulseMask is
    Port (
        clk         : in  STD_LOGIC;             -- Clock input
        reset       : in  STD_LOGIC;             -- Asynchronous reset 
        passthrough : in  signed(15 downto 0);   -- Signal to passthrough on mask high 
        divider     : in  unsigned(31 downto 0); -- Clock divider to control mask frequency
        duty        : in  unsigned(31 downto 0); -- Controls pulse width of the mask
        finalOut    : out signed(15 downto 0);   -- Output signal
        maskDAC     : out signed(15 downto 0);   -- Mask representation output for analog port
        maskDIO     : out STD_LOGIC              -- Mask representation output for DIO port
    );
end PulseMask;

architecture Behavioral of PulseMask is
    signal count   : unsigned(31 downto 0) := (others => '0');  

begin
    process(clk, reset)
    begin
        if rising_edge(clk) then
            if reset = '1' or divider = 0 or duty = 0 then -- divider or duty of 0 (i.e. initial Control values) will result in a reset condition
                count <= (others => '0');
                finalOut <= (others => '0'); -- mask is 'false', therefore set output to zero
                maskDIO <= '0';
                maskDAC <= x"8000"; -- Output Largest negative 16 bit number for pulse mask visualization
            elsif duty > divider then -- if duty cycle is higher than pulse divider, force output high at all times
                count <= (others => '0');
                finalOut <= passthrough; -- mask is 'true', therefore passthrough the input signal
                maskDIO <= '1';
                maskDAC <= x"7FFF"; -- Largest positive 16 bit number for pulse mask visualization
            elsif count >= divider - 1 then
                maskDIO <= '1';
                maskDAC <= x"7FFF"; -- Largest positive 16 bit number for pulse mask visualization
                finalOut <= passthrough; -- mask is 'true', therefore passthrough the input signal
                count <= (others => '0');
            elsif count >= duty - 1 then
                maskDIO <= '0';
                maskDAC <= x"8000"; -- Output Largest negative 16 bit number for pulse mask visualization
                finalOut <= (others => '0'); -- mask is 'false', therefore set output to zero
                count <= count + 1;
            else
                finalOut <= passthrough;
                count <= count + 1;
            end if;
        end if;
    end process;
end Behavioral;