library IEEE;
use IEEE.Std_Logic_1164.All;
use IEEE.Numeric_Std.all;

entity ArithmeticUnit is
    port (
        A : in signed(15 downto 0);
        B : in signed(15 downto 0);
        OpCode : in std_logic_vector(1 downto 0); -- 2-bit for selecting function
        Result : out signed(15 downto 0);
        Clk : in std_logic
    );
end entity;

architecture Behavioral of ArithmeticUnit is
    signal Add : signed(15 downto 0);
    signal Sub : signed(15 downto 0);
    signal Mult : signed(31 downto 0);
begin
    Add <= A + B;
    Sub <= A - B;
    Mult <= A * B;

    process(Clk) is
    begin
        if rising_edge(Clk) then
            case OpCode is
                when "00" => -- Addition
                    Result <= Add;
                when "01" => -- Subtraction
                    Result <= Sub;
                when "10" => -- Multiplication
                    Result <= resize(signed(Mult), 16);
                when others =>
                    Result <= A;
            end case;
        end if;
    end process;
end architecture;
