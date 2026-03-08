library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use work.Types.all;

entity SampleIntegrator is
    generic (
        G_NUM_SAMPLES : integer := 16;
        G_COUNTER_WIDTH : integer := 32
    );
    port (
        clk : in std_logic;
        reset : in std_logic;

        -- Inputs from BoxcarGen
        boxcar_in : in std_logic_vector(G_NUM_SAMPLES - 1 downto 0);
        sample_in : in signed16_array(0 to G_NUM_SAMPLES - 1);

        -- Outputs
        sum_out : out signed((G_COUNTER_WIDTH + 16) - 1 downto 0);
        -- Done signal is now vectorized to match input frame width
        done : out std_logic_vector(G_NUM_SAMPLES - 1 downto 0)
    );
end entity;

architecture Behavioral of SampleIntegrator is
    signal accumulator : signed((G_COUNTER_WIDTH + 16) - 1 downto 0) := (others => '0');
    signal last_boxcar_bit : std_logic := '0';

    -- Pipeline registers to break the critical path
    signal frame_sum_pipe : signed((G_COUNTER_WIDTH + 16) - 1 downto 0) := (others => '0');
    signal done_vec_pipe : std_logic_vector(G_NUM_SAMPLES - 1 downto 0) := (others => '0');
    signal window_ended_pipe : std_logic := '0';
begin

    -- Main processing loop for integration and edge detection
    process (clk)
        variable v_frame_sum : signed((G_COUNTER_WIDTH + 16) - 1 downto 0);
        variable v_done_vec : std_logic_vector(G_NUM_SAMPLES - 1 downto 0);
        variable v_window_ended : std_logic;
        variable v_prev_bit : std_logic;
    begin
        if rising_edge(clk) then
            if reset = '1' then
                accumulator <= (others => '0');
                sum_out <= (others => '0');
                last_boxcar_bit <= '0';
                done <= (others => '0');
                frame_sum_pipe <= (others => '0');
                done_vec_pipe <= (others => '0');
                window_ended_pipe <= '0';
            else
                -- Stage 1: Calculate Frame Sum and Detect Edges
                v_frame_sum := (others => '0');
                v_done_vec := (others => '0');
                v_window_ended := '0';

                -- 1. Unconditional Parallel Summation
                -- Synthesizer creates a balanced adder tree here
                -- Sums all samples in the current parallel vector
                for i in 0 to G_NUM_SAMPLES - 1 loop
                    v_frame_sum := v_frame_sum + resize(sample_in(i), (G_COUNTER_WIDTH + 16));
                end loop;

                -- 2. Per-Sample Falling Edge Detection
                -- Synchronizes the 'done' bit to the exact sample index
                -- Detects the end of a boxcar window (1 -> 0 transition)
                for i in 0 to G_NUM_SAMPLES - 1 loop
                    if i = 0 then
                        v_prev_bit := last_boxcar_bit;
                    else
                        v_prev_bit := boxcar_in(i - 1);
                    end if;

                    if v_prev_bit = '1' and boxcar_in(i) = '0' then
                        v_done_vec(i) := '1';
                        v_window_ended := '1';
                    end if;
                end loop;

                -- Update history for next cycle
                last_boxcar_bit <= boxcar_in(G_NUM_SAMPLES - 1);

                -- Pipeline Registers
                frame_sum_pipe <= v_frame_sum;
                done_vec_pipe <= v_done_vec;
                window_ended_pipe <= v_window_ended;

                -- Stage 2: Accumulation Logic (using pipelined values)
                if window_ended_pipe = '1' then
                    -- Latch total sum and reset accumulator
                    sum_out <= accumulator + frame_sum_pipe;
                    accumulator <= (others => '0');
                else
                    -- Continue accumulation
                    accumulator <= accumulator + frame_sum_pipe;
                end if;

                -- Update outputs and persistent registers
                done <= done_vec_pipe;
            end if;
        end if;
    end process;

end architecture;