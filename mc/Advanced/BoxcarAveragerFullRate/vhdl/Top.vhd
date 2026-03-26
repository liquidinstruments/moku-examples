library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.Types.all;

architecture Behavioural of CustomInstrumentInterlaced is

    constant G_NUM_SAMPLES : integer := input_interlacing_factor;
    constant G_COUNTER_WIDTH_1 : integer := 32;
    constant G_COUNTER_WIDTH_2 : integer := 32;
    constant TRG_IDX_WIDTH : integer := 4;

    signal sample_in : signed16_array(0 to G_NUM_SAMPLES - 1);
    signal trigger_in : signed16_array(0 to G_NUM_SAMPLES - 1);

    signal boxcar_sum_out : signed_sum_array(0 to G_NUM_SAMPLES - 1)((16 + G_COUNTER_WIDTH_1 + G_COUNTER_WIDTH_2) - 1 downto 0);

    signal boxcar_window : std_logic_vector(G_NUM_SAMPLES - 1 downto 0);
    signal sample_sync : signed16_array(0 to G_NUM_SAMPLES - 1);
    signal masked_sample_sync : signed16_array(0 to G_NUM_SAMPLES - 1);
    signal gain_out : signed16_array(0 to G_NUM_SAMPLES - 1);

    -- Delay line for boxcar window to match GainStage latency
    constant C_WINDOW_DELAY : integer := 5; -- Adjust this (Integrator + GainStage latency)
    type t_window_delay is array (0 to C_WINDOW_DELAY - 1) of std_logic_vector(G_NUM_SAMPLES - 1 downto 0);
    signal boxcar_window_d : t_window_delay;

begin

    -- Map input vector A to signed array
    gen_sample_in : for k in 0 to G_NUM_SAMPLES - 1 generate
        sample_in(k) <= signed(InputA(k)(15 downto 0));
    end generate gen_sample_in;

    -- Map input vector B to trigger array
    gen_trigger_in : for k in 0 to G_NUM_SAMPLES - 1 generate
        trigger_in(k) <= signed(InputB(k)(15 downto 0));
    end generate gen_trigger_in;

    -- Instantiate the main Boxcar processing wrapper
    inst_Boxcar : entity work.BoxcarWrapper
        generic map(
            G_NUM_SAMPLES => G_NUM_SAMPLES,
            G_COUNTER_WIDTH_1 => G_COUNTER_WIDTH_1,
            G_COUNTER_WIDTH_2 => G_COUNTER_WIDTH_2,
            TRG_IDX_WIDTH => TRG_IDX_WIDTH
        )
        port map(
            clk => clk,
            reset => reset,

            trigger_in => trigger_in,
            sample_in => sample_in,

            threshold => signed(Control(0)(15 downto 0)),
            boxcar_delay => unsigned(Control(1)),
            boxcar_len => unsigned(Control(2)),
            num_pulses_in => unsigned(Control(3)),

            boxcar_window => boxcar_window,
            sample_sync => sample_sync,
            masked_sample_out => masked_sample_sync,
            sum_total => boxcar_sum_out
        );

    -- Apply gain to the accumulated result
    inst_Gain : entity work.GainStage
        generic map(
            G_NUM_SAMPLES => G_NUM_SAMPLES,
            G_INPUT_WIDTH => (16 + G_COUNTER_WIDTH_1 + G_COUNTER_WIDTH_2),
            G_GAIN_INT_BITS => 16,
            G_GAIN_FRAC_BITS => 16
        )
        port map(
            clk => clk,
            reset => reset,
            gain => signed(Control(4)),
            sample_in => boxcar_sum_out,
            sample_out => gain_out
        );

    -- Output mapping process
    process (clk)
    begin
        if rising_edge(clk) then
            -- Pipeline delay for window alignment
            boxcar_window_d(0) <= boxcar_window;
            for i in 1 to C_WINDOW_DELAY - 1 loop
                boxcar_window_d(i) <= boxcar_window_d(i - 1);
            end loop;

            if Control(5)(0) = '0' then
                for k in 0 to G_NUM_SAMPLES - 1 loop
                    OutputA(k) <= signed(sample_sync(k)(15 downto 0));
                end loop;
            else
                for k in 0 to G_NUM_SAMPLES - 1 loop
                    OutputA(k) <= signed(gain_out(k));
                end loop;
            end if;

            if Control(5)(1) = '0' then
                for k in 0 to G_NUM_SAMPLES - 1 loop
                    -- Use delayed window to align with gain_out
                    if boxcar_window(k) = '1' then
                        -- if boxcar_window_d(C_WINDOW_DELAY - 1)(k) = '1' then
                        OutputB(k)(10) <= '1';
                    else
                        OutputB(k)(10) <= '0';
                    end if;
                end loop;
            else
                for k in 0 to G_NUM_SAMPLES - 1 loop
                    OutputB(k) <= signed(masked_sample_sync(k));
                end loop;
            end if;

        end if;
    end process;

end architecture;