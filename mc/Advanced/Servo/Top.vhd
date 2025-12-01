library IEEE;
use IEEE.Std_Logic_1164.all;
use IEEE.Numeric_Std.all;

library Moku;
use Moku.Support.ScaleOffset;
use Moku.Support.clip;

-- This design implements a Pulse Width Modulator which translates
-- InputA into a pulse train suitable for controlling a standard Servo.
--
-- The input can span the entire 16 bit range and is conditioned by
-- a DSP block with Controls 1 and 2.
-- Control0: gain of InputA, signed(15 downto 0)
-- Control1: offset of InputA, signed(15 downto 0)
--
-- OutputA will be a ~50Hz pulse train with the width of each pulse
-- modulated from 500usS to 2500uS depending on InputA. The output
-- amplitude is set from 0 to + MAX.
--
-- This is designed for Moku:Pro's 312.5MHz clock. The clock dividers
-- below will need to be changed when running on other platforms.
--
-- The resolution of the pulse width is 2048 increments over the
-- 2ms span.
--
-- When output to a DAC, this should be appropriate for a standard
-- servo.

architecture Behavioural of CustomWrapper is
    constant HI_LVL : signed(15 downto 0) := x"7FFF";
    constant LO_LVL : signed(15 downto 0) := x"0000";
    signal Value : signed(12 downto 0); -- The scaled input
    signal Count : unsigned(12 downto 0); -- Timer for pulse width
    signal Pulse50Hz : std_logic; -- Strobe at 50Hz to start each pulse
    signal Pulse : std_logic; -- Strobe we can count to set the width of each pulse
begin
    -- Adjust the gain and offset of InputA, and output to Value
    INPUT_SCALE: ScaleOffset
        port map (
            Clk => Clk,
            Reset => Reset,
            X => InputA,
            Scale => signed(Control0(15 downto 0)),
            Offset => signed(Control1(15 downto 0)),
            Z => Value,
            Valid => Pulse50Hz,
            OutValid => open
        );

    -- These two Counters generate out Pulse signals by dividing
    -- our system clock (312.5MHz on Moku:Pro). The divisor is
    -- 2^(EXPONENT) / Increment and is used to generate both the
    -- rate at which servo pulses are output, and the ticks within
    -- a pulse that govern the pulse width resolution.
    OSC: entity WORK.Counter
        generic map (
            EXPONENT => 24   -- 2^24 / 3 ~ 50Hz from 312.5MHz.
        )
        port map (
            Clk => Clk,
            Reset => Reset,
            Enable => '1',
            Increment => to_unsigned(3, 4),
            Strobe => Pulse50Hz
        );

    OSC2: entity WORK.Counter
        generic map (
            EXPONENT => 15 -- 2^15 / 107 gives roughly 2048 steps in a 2ms.
        )
        port map (Clk => Clk,
            Reset => Pulse50Hz,
            Enable => '1',
            Increment => to_unsigned(107, 8),
            Strobe => Pulse
        );

    -- We start this counter at the current Value and count down on
    -- each assertion of Pulse. Then stop and wait until the next Pulse50
    process(Clk) is
    begin
        if rising_edge(Clk) then
            if Pulse50Hz = '1' then
                -- Reset at ~50Hz, start at the current input Value
                -- adding 512 gives the output pulse a minimum 500uS width
                Count <= resize(unsigned(clip(Value, 11, 0)), Count'length) + 512;
            elsif Pulse = '1' and Count /= 0 then
                -- Count down on each Pulse and stop at zero
                Count <= Count - 1;
            end if;
        end if;
    end process;

    -- Finally we convert our Count value into our pulse train
    OutputA <= HI_LVL when Count /= 0 else LO_LVL;
end architecture;
