library IEEE;
use IEEE.Std_Logic_1164.all;
use IEEE.Numeric_Std.all;

-- Generates the VGA control signals, both to the display (sync signals) and for
-- the rest of the code (pixel location).
-- Instantiates the display logic that generates the image to be displayed

entity FrameBlock is
  generic(
    H_TOTAL : integer;
    V_TOTAL : integer;

    H_DISPLAY : integer;
    V_DISPLAY : integer;

    H_FRONT_PORCH : integer;
    H_SYNC_PULSE : integer;
    H_BACK_PORCH : integer;

    V_FRONT_PORCH : integer;
    V_SYNC_PULSE : integer;
    V_BACK_PORCH : integer
  );
  port(
    -- Input
    DataClock : in std_logic;
    InputData : in signed(15 downto 0);
    InputDataDownsampling : in unsigned(31 downto 0);

    -- Output
    H_Sync : out std_logic;
    V_Sync : out std_logic;
    -- Pixel Data
    DataRed : out unsigned(7 downto 0);
    DataGreen : out unsigned(7 downto 0);
    DataBlue : out unsigned(7 downto 0)
  );
end entity;

architecture Behavioural of FrameBlock is
    signal H_Counter : unsigned(15 downto 0);
    signal V_Counter : unsigned(15 downto 0);
begin
    ------------------------------------------
    -- Port maps
    U_DISPLAY_BLOCK: entity Work.DisplayBlock
        generic map (
            H_DISPLAY => H_DISPLAY,
            V_DISPLAY => V_DISPLAY
        )
        port map (
            -- Counters
            HCounter => H_Counter,
            VCounter => V_Counter,
            DataClock => DataClock,
            InputData => InputData,
            InputDataDownsampling => InputDataDownsampling,

            -- Pixel Data
            DataRed => DataRed,
            DataGreen => DataGreen,
            DataBlue => DataBlue
        );
    ------------------------------------------

    process(DataClock)
        begin
        if rising_edge(DataClock) then

            -- Counter for the current pixel location
            if (H_Counter < H_TOTAL - 1) then
                H_Counter <= H_Counter + 1;
            else
                H_Counter <= "0000000000000000";

                if (V_Counter < V_TOTAL - 1) then
                    V_Counter <= V_Counter + 1;
                else
                    V_Counter <= "0000000000000000";
                end if;
            end if;

            -- Horizontial Porch
            if H_Counter < (H_DISPLAY + H_FRONT_PORCH) OR H_Counter >= (H_DISPLAY + H_FRONT_PORCH + H_SYNC_PULSE) then
                H_Sync <= '1'; -- Active Low
            else
                H_Sync <= '0';
            end if;

            -- Vertical Porch
            if V_Counter < (V_DISPLAY + V_FRONT_PORCH) OR V_Counter >= (V_DISPLAY + V_FRONT_PORCH + V_SYNC_PULSE) then
                V_Sync <= '1'; -- Active Low
            else
                V_Sync <= '0';
            end if;

        end if;
    end process;

end architecture;
