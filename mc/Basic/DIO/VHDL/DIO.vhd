library IEEE;
use IEEE.Std_Logic_1164.all;
use IEEE.Numeric_Std.all;

-- To use this, you must configure the MCC block in the Multi-instrument Mode builder as follows:
-- MCC Slot's Input A -> DIO
-- MCC Slot's Output B -> DIO
-- DIO Pin 1-8 set as Input
-- DIO Pin 9-16 set as Output

architecture Behavioural of CustomInstrument is
    signal Count: unsigned(2 downto 0);

begin
    OutputA(0) <= InputA(8); -- Loop back Pin 9 to Pin 1
    OutputA(1) <= not InputA(9); -- Pin 2 is the inverse of Pin 10
    
    OutputA(2) <= Count(0); -- Pin 3 is a clock at 15.625MHz (Moku:Go MCC core clock is 31.25MHz)
    OutputA(3) <= Count(1); -- Pin 4 is a clock at half the rate of Pin 3
    OutputA(4) <= Count(2); -- .. and Pin 5 is half the rate again

    OutputA(5) <= InputA(10) and InputA(11); -- Logical AND
    OutputA(6) <= InputA(10) or InputA(11); -- Logical OR

    COUNTER: process(Clk)
    begin
        if rising_edge(Clk) then
            if Reset = '1' then
                Count <= (others => '0');
            else
                Count <= Count + 1;
            end if;
        end if;
    end process;
end architecture;
