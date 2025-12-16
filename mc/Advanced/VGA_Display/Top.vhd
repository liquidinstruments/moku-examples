library IEEE;
use IEEE.Std_Logic_1164.all;
use IEEE.Numeric_Std.all;

-- Defines VGA timing constants and instantiates the core logic.
-- Configure the output logic at the end of this module to change the pinout,
-- including using different colour channels.

architecture Behavioural of CustomInstrument is
    -- This is the specification to run the screen. 640*480 75fps
    CONSTANT H_TOTAL : integer := 800;
    CONSTANT V_TOTAL : integer := 525;

    CONSTANT H_DISPLAY : integer := 640;
    CONSTANT V_DISPLAY : integer := 480;

    CONSTANT H_FRONT_PORCH : integer := 16;
    CONSTANT H_SYNC_PULSE : integer := 96;
    CONSTANT H_BACK_PORCH : integer := 48;

    CONSTANT V_FRONT_PORCH : integer := 11;
    CONSTANT V_SYNC_PULSE : integer := 2;
    CONSTANT V_BACK_PORCH : integer := 32;
    ------------------------------------------
    -- Data Signals
    signal V_Sync : std_logic; -- Active LOW
    signal H_Sync : std_logic; -- Active LOW

    -- RGB signals. Is read as analog from 0-0.7V. 8 bit number meaning that 2.74mV per bit
    signal DataRed : unsigned(7 downto 0);
    signal DataGreen : unsigned(7 downto 0);
    signal DataBlue : unsigned(7 downto 0);
begin
    U_FRAME_BLOCK: entity Work.FrameBlock
        generic map (
            H_TOTAL => H_TOTAL,
            V_TOTAL => V_TOTAL,

            H_DISPLAY => H_DISPLAY,
            V_DISPLAY => V_DISPLAY,

            H_FRONT_PORCH => H_FRONT_PORCH,
            H_SYNC_PULSE => H_SYNC_PULSE,
            H_BACK_PORCH => H_BACK_PORCH,

            V_FRONT_PORCH => V_FRONT_PORCH,
            V_SYNC_PULSE => V_SYNC_PULSE,
            V_BACK_PORCH => V_BACK_PORCH
        )
        port map (
            DataClock => Clk, -- Running at 31.25MHz
            H_Sync => H_Sync,
            V_Sync => V_Sync,
            InputDataDownsampling => unsigned(Control(2)),
            InputData => InputA,
            -- Pixel Data
            DataRed => DataRed,
            DataGreen => DataGreen,
            DataBlue => DataBlue
        );

    -- Output Assignments
    OutputA(11 downto 0) <= signed(DataRed & "1111");
    OutputB(11 downto 0) <= signed(DataBlue & "1111");
    OutputC(0) <= H_Sync;
    OutputC(1) <= V_Sync;

end architecture;
