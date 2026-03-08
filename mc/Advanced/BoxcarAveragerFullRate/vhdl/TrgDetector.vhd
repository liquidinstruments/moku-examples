library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use IEEE.math_real.all;
use work.Types.all;

entity TrgDetector is
    generic (G_NUM_SAMPLES : integer := 16;
                                        TRG_IDX_WIDTH : integer := 4
                                        );
    port (
        clk : in std_logic;
        reset : in std_logic;
        threshold : in signed(15 downto 0);
        sample_in : in signed16_array(0 to G_NUM_SAMPLES - 1);
        trg_pulse_mask : out std_logic_vector(G_NUM_SAMPLES - 1 downto 0);
        trg_idx : out unsigned(TRG_IDX_WIDTH - 1 downto 0)
    );
end entity;

architecture Behavioral of TrgDetector is
    -- Stores the last sample from the previous clock cycle
    signal last_sample_prev_frame : signed(15 downto 0);
begin
    process (clk)
        variable v_pulse : std_logic_vector(G_NUM_SAMPLES - 1 downto 0);
        variable v_found : boolean;
        variable v_trg_idx : unsigned(TRG_IDX_WIDTH - 1 downto 0);
        variable v_prev_comp : signed(15 downto 0);
    begin
        if rising_edge(clk) then
            v_pulse := (others => '0');
            v_trg_idx := (others => '0');

            if reset = '1' then
                last_sample_prev_frame <= (others => '0');
                trg_pulse_mask <= (others => '0');
            else
                v_found := false;

                for i in 0 to G_NUM_SAMPLES - 1 loop
                    -- Determine which sample to compare against
                    if i = 0 then
                        v_prev_comp := last_sample_prev_frame;
                    else
                        v_prev_comp := sample_in(i - 1);
                    end if;

                    -- Trigger Condition: Current > Threshold AND Previous <= Threshold
                    -- Only detect the first occurrence per frame to stay consistent
                    if (sample_in(i) > threshold) and (v_prev_comp <= threshold) and (not v_found) then
                        v_pulse(i) := '1';
                        v_found := true;
                        v_trg_idx := to_unsigned(i, TRG_IDX_WIDTH);
                    end if;
                end loop;

                -- Store the last sample of this frame for the next clock cycle's first comparison
                last_sample_prev_frame <= sample_in(G_NUM_SAMPLES - 1);
                trg_pulse_mask <= v_pulse;
                trg_idx <= v_trg_idx;
            end if;
        end if;
    end process;
end architecture;