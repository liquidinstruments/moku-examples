library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use work.Types.all;

entity BoxcarWrapper is
    generic (
        G_NUM_SAMPLES : integer := 4;
        G_COUNTER_WIDTH_1 : integer := 32; -- Width for SampleIntegrator
        G_COUNTER_WIDTH_2 : integer := 32; -- Width for Integrator2nd pulse count
        TRG_IDX_WIDTH : integer := 4
    );
    port (
        clk : in std_logic;
        reset : in std_logic;

        -- Data Input
        trigger_in : in signed16_array(0 to G_NUM_SAMPLES - 1);
        sample_in : in signed16_array(0 to G_NUM_SAMPLES - 1);

        -- Trigger Configuration
        threshold : in signed(15 downto 0);

        -- Boxcar Configuration
        boxcar_len : in unsigned(G_COUNTER_WIDTH_1 - 1 downto 0);
        boxcar_delay : in unsigned(G_COUNTER_WIDTH_1 - 1 downto 0);

        -- Integrator Configuration
        num_pulses_in : in unsigned(G_COUNTER_WIDTH_2 - 1 downto 0);

        -- Outputs
        sum_total : out signed_sum_array(0 to G_NUM_SAMPLES - 1)((16 + G_COUNTER_WIDTH_1 + G_COUNTER_WIDTH_2) - 1 downto 0);
        boxcar_window : out std_logic_vector(G_NUM_SAMPLES - 1 downto 0);
        sample_sync : out signed16_array(0 to G_NUM_SAMPLES - 1);
        masked_sample_out : out signed16_array(0 to G_NUM_SAMPLES - 1);
        total_ready : out std_logic_vector(G_NUM_SAMPLES - 1 downto 0)
    );
end entity;

architecture Behavioral of BoxcarWrapper is

    -- Internal Signals
    signal trg_pulse_mask : std_logic_vector(G_NUM_SAMPLES - 1 downto 0);
    signal trg_idx : unsigned(TRG_IDX_WIDTH - 1 downto 0);

    signal boxcar_out : std_logic_vector(G_NUM_SAMPLES - 1 downto 0);
    signal masked_samples : signed16_array(0 to G_NUM_SAMPLES - 1);
    signal sample_in_d : signed16_array(0 to G_NUM_SAMPLES - 1);
    signal sample_in_d_d : signed16_array(0 to G_NUM_SAMPLES - 1);

    signal sum_out_sample : signed((G_COUNTER_WIDTH_1 + 16) - 1 downto 0);
    signal done_vec_sample : std_logic_vector(G_NUM_SAMPLES - 1 downto 0);

begin

    -- Detects trigger conditions in the input stream
    inst_TrgDetector : entity work.TrgDetector
        generic map(
            G_NUM_SAMPLES => G_NUM_SAMPLES,
            TRG_IDX_WIDTH => TRG_IDX_WIDTH
        )
        port map(
            clk => clk,
            reset => reset,
            threshold => threshold,
            sample_in => trigger_in,
            trg_pulse_mask => trg_pulse_mask,
            trg_idx => trg_idx
        );

    -- Delay sample input to align with trigger detection latency if necessary
    process (clk)
    begin
        if rising_edge(clk) then
            sample_in_d <= sample_in;
            sample_in_d_d <= sample_in_d;
        end if;
    end process;

    -- Generates the boxcar window mask based on triggers
    inst_BoxcarGen : entity work.BoxcarGen
        generic map(
            G_NUM_SAMPLES => G_NUM_SAMPLES,
            G_COUNTER_WIDTH => G_COUNTER_WIDTH_1,
            TRG_IDX_WIDTH => TRG_IDX_WIDTH
        )
        port map(
            clk => clk,
            reset => reset,
            max_count_in => boxcar_len,
            trg_delay_in => boxcar_delay,
            trg_mask_in => trg_pulse_mask,
            trg_idx_in => trg_idx,
            sample_in => sample_in_d,
            boxcar_out => boxcar_out,
            sample_out => masked_samples
        );
    boxcar_window <= boxcar_out;
    sample_sync <= sample_in_d_d;
    masked_sample_out <= masked_samples;

    -- Integrates samples within the boxcar window
    inst_SampleIntegrator : entity work.SampleIntegrator
        generic map(
            G_NUM_SAMPLES => G_NUM_SAMPLES,
            G_COUNTER_WIDTH => G_COUNTER_WIDTH_1
        )
        port map(
            clk => clk,
            reset => reset,
            boxcar_in => boxcar_out,
            sample_in => masked_samples,
            sum_out => sum_out_sample,
            done => done_vec_sample
        );

    -- Accumulates multiple boxcar windows (pulses) into a final total
    inst_Integrator2nd : entity work.Integrator2nd
        generic map(
            G_NUM_SAMPLES => G_NUM_SAMPLES,
            G_COUNTER_WIDTH_1 => G_COUNTER_WIDTH_1,
            G_COUNTER_WIDTH_2 => G_COUNTER_WIDTH_2
        )
        port map(
            clk => clk,
            reset => reset,
            num_pulses_in => num_pulses_in,
            sum_in => sum_out_sample,
            done_in => done_vec_sample,
            sum_total => sum_total,
            total_ready => total_ready
        );

end architecture;