library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;
use work.Types.all;

entity GainStage is
    generic (
        G_NUM_SAMPLES : integer := 16;
        G_INPUT_WIDTH : integer := 16;
        G_GAIN_INT_BITS : integer := 8;
        G_GAIN_FRAC_BITS : integer := 8
    );
    port (
        clk : in std_logic;
        reset : in std_logic;

        -- Gain input: Fixed-point value Q(Int).(Frac)
        gain : in signed((G_GAIN_INT_BITS + G_GAIN_FRAC_BITS) - 1 downto 0);

        sample_in : in signed_sum_array(0 to G_NUM_SAMPLES - 1)(G_INPUT_WIDTH - 1 downto 0);
        sample_out : out signed16_array(0 to G_NUM_SAMPLES - 1)
    );
end entity;

architecture Behavioral of GainStage is
    constant C_GAIN_WIDTH : integer := G_GAIN_INT_BITS + G_GAIN_FRAC_BITS;
    -- Product width = Gain Width + Gain Width (Input is saturated to Gain Width)
    constant C_PRODUCT_WIDTH : integer := C_GAIN_WIDTH + C_GAIN_WIDTH;

    -- Indices for extracting the 16-bit integer part
    -- We discard the fractional bits (G_GAIN_FRAC_BITS)
    constant C_LSB_IDX : integer := G_GAIN_FRAC_BITS;
    constant C_MSB_IDX : integer := C_LSB_IDX + 15;

    -- Pipeline registers
    signal sample_sat_stage1 : signed_sum_array(0 to G_NUM_SAMPLES - 1)(C_GAIN_WIDTH - 1 downto 0);
    signal gain_stage1 : signed(C_GAIN_WIDTH - 1 downto 0);
    signal product_stage2 : signed_sum_array(0 to G_NUM_SAMPLES - 1)(C_PRODUCT_WIDTH - 1 downto 0);

begin

    -- Pipeline Stage 1: Input Saturation and Input Registering
    -- Saturates the wide input sample to the width of the gain value,
    -- then registers the result and the gain for the next stage.
    process (clk)
        variable v_sample_sat : signed(C_GAIN_WIDTH - 1 downto 0);
    begin
        if rising_edge(clk) then
            if reset = '1' then
                sample_sat_stage1 <= (others => (others => '0'));
                gain_stage1 <= (others => '0');
            else
                for i in 0 to G_NUM_SAMPLES - 1 loop
                    -- Saturate Input to Gain Width
                    v_sample_sat := resize(sample_in(i), C_GAIN_WIDTH);

                    if G_INPUT_WIDTH > C_GAIN_WIDTH then
                        if resize(v_sample_sat, G_INPUT_WIDTH) /= sample_in(i) then
                            if sample_in(i)(G_INPUT_WIDTH - 1) = '0' then
                                v_sample_sat := (others => '1');
                                v_sample_sat(C_GAIN_WIDTH - 1) := '0'; -- Max Positive
                            else
                                v_sample_sat := (others => '0');
                                v_sample_sat(C_GAIN_WIDTH - 1) := '1'; -- Max Negative
                            end if;
                        end if;
                    end if;
                    sample_sat_stage1(i) <= v_sample_sat;
                end loop;
                gain_stage1 <= gain;
            end if;
        end if;
    end process;

    -- Pipeline Stage 2: Multiplication
    -- Multiplies the registered sample and gain, and registers the product.
    process (clk)
    begin
        if rising_edge(clk) then
            if reset = '1' then
                product_stage2 <= (others => (others => '0'));
            else
                for i in 0 to G_NUM_SAMPLES - 1 loop
                    product_stage2(i) <= sample_sat_stage1(i) * gain_stage1;
                end loop;
            end if;
        end if;
    end process;

    -- Pipeline Stage 3: Output Saturation and Assignment
    -- Takes the wide product, discards fractional bits, saturates to 16 bits,
    -- and assigns to the output port.
    process (clk)
        variable v_check : signed(C_PRODUCT_WIDTH - 1 downto C_MSB_IDX);
    begin
        if rising_edge(clk) then
            if reset = '1' then
                sample_out <= (others => (others => '0'));
            else
                for i in 0 to G_NUM_SAMPLES - 1 loop
                    if C_PRODUCT_WIDTH - 1 > C_MSB_IDX then
                        v_check := product_stage2(i)(C_PRODUCT_WIDTH - 1 downto C_MSB_IDX);

                        if (v_check = (v_check'range => '0')) or (v_check = (v_check'range => '1')) then
                            sample_out(i) <= product_stage2(i)(C_MSB_IDX downto C_LSB_IDX);
                        else
                            if product_stage2(i)(C_PRODUCT_WIDTH - 1) = '0' then
                                sample_out(i) <= x"7FFF"; -- Max Positive
                            else
                                sample_out(i) <= x"8000"; -- Max Negative
                            end if;
                        end if;
                    else
                        sample_out(i) <= product_stage2(i)(C_MSB_IDX downto C_LSB_IDX);
                    end if;
                end loop;
            end if;
        end if;
    end process;

end architecture;