LIBRARY IEEE;
USE IEEE.Std_Logic_1164.ALL;
USE IEEE.Numeric_Std.ALL;

-- This code here stores the data for each point coming off an oscilloscope.
-- The rate of data entry is determined by the InputDataDownsampling value, which acts as a direct decimator factor on the incoming data.

-- The data is double-buffered for triggering, with a separate output buffer.

-- Trigger is set for Rising edge

entity DataBlock is
    generic (
        H_DISPLAY : integer;
        V_DISPLAY : integer
    );
    port (
        -- Input
        DataClock : in std_logic;
        InputData : in signed(15 downto 0);
        InputDataDownsampling : in unsigned(31 downto 0);
        HCounter : in unsigned(15 downto 0);
        -- Output
        OutputValue : out signed(8 downto 0)
    );
end entity;

architecture Behavioural of DataBlock is
    type DataArrayScreen is array (0 to H_DISPLAY) of signed(8 downto 0); -- Defining the array
    signal R_Data : DataArrayScreen; -- Display Read Array
    signal W1_Data : DataArrayScreen; -- Display Write1 Array
    signal W2_Data : DataArrayScreen; -- Display Write2 Array

    signal ArrayAddressCounter : unsigned(15 downto 0); -- Array Wrapping
    signal InputDataClock : unsigned(31 downto 0); -- Downsampling clock
    signal InputDataNorm : signed(15 downto 0);

    signal CurrentBuffer : std_logic;
    signal Trig : std_logic;
    signal InputDataPrevious : signed(15 downto 0);
begin

    process(DataClock)
    begin
        if rising_edge(DataClock) then
            ------------------------------------------
            -- We need to limit the H_Counter to the display array to read data in Array
            if HCounter < H_DISPLAY - 1 then
                OutputValue <= R_Data(to_integer(HCounter)); -- Reading new data
            else
                -- Once we leave the display area it's safe to update the output buffer. We do this with a huge latch in one clock cycle, we have the gates for it!
                if CurrentBuffer = '1' then
                    for I in 0 to H_DISPLAY loop
                        R_Data(I) <= W1_Data(I); -- W1 map
                    end loop;
                else
                    for I in 0 to H_DISPLAY loop
                        R_Data(I) <= W2_Data(I); -- W2 map
                    end loop;
                end if;
            end if;
            ------------------------------------------

            -- Variable period between samples
            if InputDataClock < InputDataDownsampling then
                InputDataClock <= InputDataClock + 1;
            else
                InputDataClock <= (others => '0');
                InputDataPrevious <= InputData; -- Writing to previous for rising edge.

                if Trig = '1' then -- Clock cycles for data storage for buf1

                    -- Wrapping Array Addresses
                    if (ArrayAddressCounter < H_DISPLAY - 1) then
                        ArrayAddressCounter <= ArrayAddressCounter + 1;
                    else
                        ArrayAddressCounter <= (others => '0');

                        -- Only does rising edge when signal is fast enough that you cannot see it via scanning.
                        if InputDataDownsampling < 30000 then
                            CurrentBuffer <= not(CurrentBuffer); -- Cycling buffer (once per frame)
                            Trig <= '0'; -- Reset trig (once per frame).
                        else
                            CurrentBuffer <= '1'; -- Never resets, only runs on one of the registers.
                            Trig <= '1';
                        end if;

                    end if;

                    -- Begin trigger process
                    InputDataNorm <= shift_left(InputData, 4);
                    if CurrentBuffer = '1' then
                        W1_Data(to_integer(ArrayAddressCounter)) <= InputDataNorm(15 downto 7); -- Writing new data
                    else
                        W2_Data(to_integer(ArrayAddressCounter)) <= InputDataNorm(15 downto 7); -- Writing new data
                    end if;

                else
                    -- When Trig='0' we check to see if the rising edge has started. If so, we begin measuring the data.
                    if InputData > 0 and InputDataPrevious < 0 then
                        Trig <= '1';
                    end if;

                end if; -- Wrapping Trig if
            end if; -- Variable period if
        end if; -- rising edge if
    end process;

end architecture;
