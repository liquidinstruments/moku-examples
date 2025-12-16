library IEEE;
use IEEE.Std_Logic_1164.all;
use IEEE.Numeric_Std.all;

-- A Counter provides an "event timer" by diving inputs events
-- by 2^EXPONENT / Increment
entity Counter is
    generic (
        EXPONENT : positive := 8;
        PHASE90 : boolean := false
    );
    port (
        Clk : in std_logic;
        Reset : in std_logic;
        Enable : in std_logic; -- Drive to '1' to divide Clk
        Increment : in unsigned;
        Strobe : out std_logic -- Output events
    );
end entity;

architecture Behavioural of Counter is
    signal Count : unsigned(EXPONENT downto 0);
begin

    assert Increment'length <= Count'length severity FAILURE;

    process(Clk) is
    begin
        if rising_edge(Clk) then
            if Reset = '1' then
                Count <= (others => '0');
                if PHASE90 then
                    Count(EXPONENT - 1) <= '1';
                end if;
            elsif Enable = '1' then
                -- Trim the MSB but allow overflow into it. This gives a single Clk cycle
                -- output pulse on Strobe.
                Count <= resize(Count(Count'left - 1 downto 0), Count'length) + Increment;
            else
                -- Prevent output longer the a single Clk cycle
                Count(Count'left) <= '0';
            end if;
        end if;
    end process;

    -- Just use the MSB of Count overflowing as our event
    Strobe <= Count(Count'left);

end architecture;
