library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use work.Types.all;

entity BoxcarGen is
    generic (
        G_NUM_SAMPLES : integer := 16;
        G_COUNTER_WIDTH : integer := 32;
        TRG_IDX_WIDTH : integer := 4
    );
    port (
        clk : in std_logic;
        reset : in std_logic;

        -- Configuration
        max_count_in : in unsigned(G_COUNTER_WIDTH - 1 downto 0);
        trg_delay_in : in unsigned(G_COUNTER_WIDTH - 1 downto 0);

        -- Inputs
        trg_mask_in : in std_logic_vector(G_NUM_SAMPLES - 1 downto 0);
        trg_idx_in : in unsigned(TRG_IDX_WIDTH - 1 downto 0);
        sample_in : in signed16_array(0 to G_NUM_SAMPLES - 1);

        -- Outputs
        boxcar_out : out std_logic_vector(G_NUM_SAMPLES - 1 downto 0);
        sample_out : out signed16_array(0 to G_NUM_SAMPLES - 1)
    );
end entity;

architecture Behavioral of BoxcarGen is
    signal counter : unsigned(G_COUNTER_WIDTH - 1 downto 0) := (others => '1');
    constant ALL_OFF : std_logic_vector(G_NUM_SAMPLES - 1 downto 0) := (others => '0');
begin
    process (clk)
        variable v_total_limit : unsigned(G_COUNTER_WIDTH - 1 downto 0);
        variable v_curr_pos : unsigned(G_COUNTER_WIDTH - 1 downto 0);
        variable v_start_idx : integer range 0 to G_NUM_SAMPLES - 1;
        variable v_current_counter_base : unsigned(G_COUNTER_WIDTH - 1 downto 0);
        variable v_boxcar_mask : std_logic_vector(G_NUM_SAMPLES - 1 downto 0);
        variable v_trg_active : std_logic;
        variable v_base_count : unsigned(G_COUNTER_WIDTH - 1 downto 0);
    begin
        if rising_edge(clk) then
            v_boxcar_mask := (others => '0');

            if reset = '1' then
                counter <= (others => '1');
                boxcar_out <= (others => '0');
                sample_out <= (others => (others => '0'));
            else
                v_total_limit := trg_delay_in + max_count_in;

                --------------------------------------------------------------------------------
                -- 1. Trigger Activation and Counter Update
                --------------------------------------------------------------------------------
                -- If any bit in the mask is high, we treat this frame as a "Trigger Frame"
                if trg_mask_in /= ALL_OFF then
                    v_trg_active := '1';
                else
                    v_trg_active := '0';
                end if;

                -- Update the registered counter for the NEXT frame
                if v_trg_active = '1' then
                    -- The next frame will start counting from the remaining samples in this frame
                    -- e.g., if 8 samples total and trg is at index 2, 6 samples pass this cycle.
                    counter <= to_unsigned(G_NUM_SAMPLES - to_integer(trg_idx_in), G_COUNTER_WIDTH);
                    v_base_count := (others => '0'); -- Base is 0 because we calculate relative to trg_idx_in
                elsif counter > 0 and counter < v_total_limit then
                    counter <= counter + to_unsigned(G_NUM_SAMPLES, G_COUNTER_WIDTH);
                    v_base_count := counter;
                else
                    -- Idle or Finished state
                    v_base_count := counter;
                end if;

                --------------------------------------------------------------------------------
                -- 2. Parallel Masking and Data Transfer
                --------------------------------------------------------------------------------
                for i in 0 to G_NUM_SAMPLES - 1 loop
                    -- Use parallel subtraction to find the position relative to the trigger
                    if v_trg_active = '1' then
                        if i >= to_integer(trg_idx_in) then
                            v_curr_pos := to_unsigned(i - to_integer(trg_idx_in), G_COUNTER_WIDTH);
                        else
                            -- Sample occurred before the trigger in the same frame
                            v_curr_pos := (others => '1'); -- Max value (Out of Bounds)
                        end if;
                    else
                        -- Incrementing from the previous frame's end point
                        v_curr_pos := v_base_count + to_unsigned(i, G_COUNTER_WIDTH);
                    end if;

                    -- Window Masking Logic
                    if (v_curr_pos >= trg_delay_in) and (v_curr_pos < v_total_limit) then
                        v_boxcar_mask(i) := '1';
                        sample_out(i) <= sample_in(i);
                    else
                        v_boxcar_mask(i) := '0';
                        sample_out(i) <= (others => '0');
                    end if;
                end loop;

                boxcar_out <= v_boxcar_mask;
            end if;
        end if;
    end process;
end architecture;