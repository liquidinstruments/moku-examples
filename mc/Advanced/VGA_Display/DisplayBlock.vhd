library IEEE;
use IEEE.Std_Logic_1164.all;
use IEEE.Numeric_Std.all;

-- The embedded "Data Block" implements the core Oscilloscope function and outputs the voltage recorded at the current horizontal
-- position. This block converts that to colour values of specific pixels for display on the screen.

entity DisplayBlock is
  generic(
    H_DISPLAY : integer;
    V_DISPLAY : integer
  );
  port(
    -- Inputs
    -- Counters
    HCounter : in unsigned(15 downto 0);
    VCounter : in unsigned(15 downto 0);
    DataClock : in std_logic;
    -- Input Data
    InputData : in signed(15 downto 0);
    InputDataDownsampling : in unsigned(31 downto 0);

    -- Outputs
    -- Pixel Data
    DataRed : out unsigned(7 downto 0);
    DataGreen : out unsigned(7 downto 0);
    DataBlue : out unsigned(7 downto 0)
  );
end entity;

architecture Behavioural of DisplayBlock is
    signal OutputValue : signed(8 downto 0);
    signal OutputValueAdjusted : unsigned(8 downto 0);
begin
    U_DATA_BLOCK: entity Work.DataBlock
        generic map (
            H_DISPLAY => H_DISPLAY,
            V_DISPLAY => V_DISPLAY
            )
        port map (
            DataClock => DataClock,
            InputData => InputData,
            InputDataDownsampling => InputDataDownsampling,
            HCounter => HCounter,

            OutputValue => OutputValue
        );

    process(HCounter, VCounter)
        begin

        if (HCounter < H_DISPLAY - 1) and (VCounter < V_DISPLAY - 1) then -- Wrapping the counter to display values only (excludes porches etc.)

            OutputValueAdjusted <= unsigned(240 - OutputValue); -- Setting the centre of the moniter as 0

            -- Each data point is rendered as a 16-pixel high line. A more advanced implementation would trace a line between
            -- adjacent points.
            if OutputValueAdjusted > VCounter - "1000" and OutputValueAdjusted < VCounter + "1000" then
            DataRed <= "11111111";
            DataBlue <= "11111111";
            DataGreen <= "11111111";
            else
            DataRed <= "01111111"; -- No signal detected
            DataBlue <= "00000000";
            DataGreen <= "00000000";
            end if;

        else
            DataRed <= "00000000"; -- When not in the Display area, no need to output anything.
            DataBlue <= "00000000";
            DataGreen <= "00000000";
        end if;

    end process;

end architecture;
