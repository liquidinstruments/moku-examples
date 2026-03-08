library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use work.Types.all;

entity Integrator2nd is
    generic (
        G_NUM_SAMPLES : integer := 4;
        G_COUNTER_WIDTH_1 : integer := 32; -- From SampleIntegrator
        G_COUNTER_WIDTH_2 : integer := 16 -- For this pulse-count integrator
    );
    port (
        clk : in std_logic;
        reset : in std_logic;

        -- Control input: Number of pulses to integrate before done
        num_pulses_in : in unsigned(G_COUNTER_WIDTH_2 - 1 downto 0);

        -- Inputs from SampleIntegrator
        sum_in : in signed((G_COUNTER_WIDTH_1 + 16) - 1 downto 0);
        done_in : in std_logic_vector(G_NUM_SAMPLES - 1 downto 0);

        -- Outputs
        -- Total width = 16 (bits) + G1 (growth) + G2 (growth)
        sum_total : out signed_sum_array(0 to G_NUM_SAMPLES - 1)((16 + G_COUNTER_WIDTH_1 + G_COUNTER_WIDTH_2) - 1 downto 0);
        total_ready : out std_logic_vector(G_NUM_SAMPLES - 1 downto 0)
    );
end entity;

architecture Behavioral of Integrator2nd is
    -- Internal signal width calculation
    constant C_TOTAL_WIDTH : integer := 16 + G_COUNTER_WIDTH_1 + G_COUNTER_WIDTH_2;

    signal accumulator : signed(C_TOTAL_WIDTH - 1 downto 0) := (others => '0');
    signal pulse_count : unsigned(G_COUNTER_WIDTH_2 - 1 downto 0) := (others => '0');
    signal last_sum_total : signed(C_TOTAL_WIDTH - 1 downto 0) := (others => '0');

    -- Pipeline registers for timing closure
    signal sum_in_d : signed((G_COUNTER_WIDTH_1 + 16) - 1 downto 0) := (others => '0');
    signal done_in_d : std_logic_vector(G_NUM_SAMPLES - 1 downto 0) := (others => '0');
begin

    process (clk)
        variable v_acc : signed(C_TOTAL_WIDTH - 1 downto 0);
        variable v_next_count : unsigned(G_COUNTER_WIDTH_2 - 1 downto 0);
        variable v_done_active : std_logic;
        variable v_current_out : signed(C_TOTAL_WIDTH - 1 downto 0);
        variable v_batch_done : boolean;
    begin
        if rising_edge(clk) then
            -- Input pipeline
            sum_in_d <= sum_in;
            done_in_d <= done_in;

            -- Default assignments for variables
            v_acc := accumulator;
            v_next_count := pulse_count;
            v_done_active := '0';
            v_current_out := last_sum_total;
            v_batch_done := false;
            total_ready <= (others => '0');

            if reset = '1' then
                accumulator <= (others => '0');
                pulse_count <= (others => '0');
                sum_total <= (others => (others => '0'));
                total_ready <= (others => '0');
                last_sum_total <= (others => '0');
                sum_in_d <= (others => '0');
                done_in_d <= (others => '0');
            else
                -- Check if any bit in done_in_d is high
                for i in 0 to G_NUM_SAMPLES - 1 loop
                    if done_in_d(i) = '1' then
                        v_done_active := '1';
                    end if;
                end loop;

                -- Process only when a completed pulse is signaled
                if v_done_active = '1' then
                    -- 1. Accumulate the new pulse area
                    v_acc := v_acc + resize(sum_in_d, C_TOTAL_WIDTH);

                    -- 2. Check for batch completion
                    -- Optimization: Check pulse_count directly to avoid adder in critical path
                    if pulse_count >= (num_pulses_in - 1) then
                        v_batch_done := true;
                    else
                        v_next_count := pulse_count + 1;
                    end if;
                end if;

                -- 3. Output generation loop with inheritance
                for i in 0 to G_NUM_SAMPLES - 1 loop
                    if done_in_d(i) = '1' and v_batch_done then
                        v_current_out := v_acc;
                        total_ready(i) <= '1';
                    end if;
                    sum_total(i) <= v_current_out;
                end loop;

                if v_done_active = '1' and v_batch_done then
                    v_acc := (others => '0'); -- Reset for next batch
                    pulse_count <= (others => '0');
                else
                    accumulator <= v_acc;
                    pulse_count <= v_next_count;
                end if;

                -- Update registers
                last_sum_total <= v_current_out;
            end if;
        end if;
    end process;

end architecture;