library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

--------------------------------------------------------------------------------
-- Entity Architecture: rtl of CustomInstrumentInterlaced
--
-- Description:
--   Multi-lane, frame-synchronized signal averager for Moku Cloud Compile.
--   Captures frames of 128 addresses (128 * INPUT_INTERLACING_FACTOR samples)
--   upon detecting a rising zero crossing on trigger input (InputB).
--   Accumulates repeated frames into an on-chip dual-port Block RAM (accum_mem),
--   scales the result by a fixed-point scale factor, saturates to signed 16 bits,
--   and streams the normalized average to OutputA with an activity flag on OutputB.
--
-- Pipeline Latency:
--   Strictly 5 clock cycles from RAM read address generation to OutputA/OutputB:
--     Cycle 1: RAM address issued (memory_address_reg)
--     Cycle 2: RAM synchronous read data emerges (memory_read_data_reg, state_d1_reg)
--     Cycle 3: RMW vector accumulation & writeback (memory_read_data_d1_reg, state_d2_reg)
--     Cycle 4: DSP fixed-point multiplication (scaled_sample_reg, state_d3_reg)
--     Cycle 5: 16-bit saturation & clamped output (output_a_reg, state_d4_reg)
--------------------------------------------------------------------------------
architecture rtl of CustomInstrumentInterlaced is

    -- Frame memory dimensions and arithmetic bit widths.
    constant C_FRAME_SIZE   : integer := 128; -- Total RAM addresses per frame (128 * M samples).
    constant C_ADDR_WIDTH   : integer := 7;   -- 7-bit address bus for 128 memory words (2^7 = 128).
    constant C_ACC_WIDTH    : integer := 28;  -- 28-bit accumulator width (prevents overflow for <= 65536 frames).
    constant C_OUTPUT_WIDTH : integer := 16;  -- Bitwidth of OutputA (platform sample / DAC port width).
    constant C_SCALE_WIDTH  : integer := 16;  -- Scale factor width (Q0.[C_SCALE_WIDTH] format).
    constant C_PROD_WIDTH   : integer := C_ACC_WIDTH + C_SCALE_WIDTH; -- DSP product width.

    -- Interlaced parallel array types parameterized by top-level platform generics.
    type t_sample_array  is array (0 to INPUT_INTERLACING_FACTOR - 1) of signed(C_OUTPUT_WIDTH - 1 downto 0);
    type t_accum_array   is array (0 to INPUT_INTERLACING_FACTOR - 1) of signed(C_ACC_WIDTH - 1 downto 0);
    type t_product_array is array (0 to OUTPUT_INTERLACING_FACTOR - 1) of signed(C_PROD_WIDTH - 1 downto 0);

    -- Acquisition finite-state machine (FSM) states:
    --   IDLE    : Armed and waiting for a rising zero crossing on InputB.
    --   DELAY   : Optional programmable holdoff delay countdown before capture starts.
    --   CAPTURE : Streaming addresses 0..127, accumulating incoming samples into RAM.
    --   READ    : Streaming addresses 0..127 without accumulation (pause limit reached).
    type t_state is (IDLE, DELAY, CAPTURE, READ);

    -- Unpacked host control registers:
    signal delay_cycles      : unsigned(31 downto 0); -- Control(0): Post-trigger delay in clock cycles.
    signal scale_factor      : signed(C_SCALE_WIDTH - 1 downto 0); -- Control(1): Q0.[C_SCALE_WIDTH] scale factor.
    signal frame_limit       : unsigned(31 downto 0); -- Control(2): Frame count limit before capture pauses.
    signal pre_leveling_bias : signed(15 downto 0);   -- Control(3): 16-bit DC bias subtraction offset for InputA.
    signal reset_cmd         : std_logic;             -- Control(4)(0): Software master reset strobe.

    -- Input sampling and alignment pipeline registers:
    signal input_a_reg                 : t_sample_array; -- Stage 0: Synchronously registered InputA lanes.
    signal input_a_d1_reg              : t_sample_array; -- Stage 1: 1-cycle delayed InputA for barrel shifter.
    signal input_a_d2_reg              : t_sample_array; -- Stage 2: Trigger-aligned samples (lane 0 = trigger).
    signal input_b_reg                 : t_sample_array; -- Stage 0: Synchronously registered InputB lanes.
    signal input_b_previous_sample_reg : signed(15 downto 0); -- Highest lane (M-1) from previous clock cycle.

    -- Trigger detection and FSM control signals:
    signal state_reg                 : t_state := IDLE; -- Current capture FSM state.
    signal trigger_found             : std_logic;       -- Combinational flag: '1' when rising crossing detected.
    signal trigger_offset            : integer range 0 to INPUT_INTERLACING_FACTOR - 1; -- Crossing lane index (0..M-1).
    signal active_trigger_offset_reg : integer range 0 to INPUT_INTERLACING_FACTOR - 1; -- Latched offset for frame.

    signal delay_counter_reg  : unsigned(31 downto 0);          -- Down-counter for post-trigger holdoff delay.
    signal frame_count_reg    : unsigned(31 downto 0);          -- Number of completed accumulated frames.
    signal memory_address_reg : unsigned(C_ADDR_WIDTH - 1 downto 0); -- RAM read address pointer (0..127).
    signal capture_paused_reg : std_logic;                      -- '1' when frame_count_reg >= frame_limit.

    -- Component declaration for shared-variable Simple Dual-Port Block RAM:
    component simple_dual_one_clock is
        generic(
            ADDR_WIDTH : integer := 7;
            DATA_WIDTH : integer := 32
        );
        port(
            clk   : in  std_logic;
            ena   : in  std_logic;
            enb   : in  std_logic;
            wea   : in  std_logic;
            addra : in  std_logic_vector(ADDR_WIDTH - 1 downto 0);
            addrb : in  std_logic_vector(ADDR_WIDTH - 1 downto 0);
            dia   : in  std_logic_vector(DATA_WIDTH - 1 downto 0);
            dob   : out std_logic_vector(DATA_WIDTH - 1 downto 0)
        );
    end component simple_dual_one_clock;

    -- Dual-port synchronous Block RAM interface signals:
    type t_bram_data_array is array (0 to INPUT_INTERLACING_FACTOR - 1) of std_logic_vector(C_ACC_WIDTH - 1 downto 0);
    signal bram_dia   : t_bram_data_array;
    signal bram_dob   : t_bram_data_array;
    signal bram_wea   : std_logic;
    signal bram_addra : std_logic_vector(C_ADDR_WIDTH - 1 downto 0);
    signal bram_addrb : std_logic_vector(C_ADDR_WIDTH - 1 downto 0);

    signal memory_read_data_reg    : t_accum_array; -- Stage 1: BRAM synchronous read output latch.
    signal memory_read_data_d1_reg : t_accum_array; -- Stage 2: Latched read data for vector adder and DSP.

    -- Datapath and control pipeline alignment registers:
    --   memory_address_d1_reg, d2_reg delay the write address by 2 cycles (A_write = A_read - 2).
    --   state_d1_reg .. state_d4_reg track the FSM state in lockstep with each processing stage.
    --   is_first_frame_reg delays the overwrite flag to match memory write latency.
    signal memory_address_d1_reg, memory_address_d2_reg : unsigned(C_ADDR_WIDTH - 1 downto 0);
    signal state_d1_reg, state_d2_reg, state_d3_reg, state_d4_reg : t_state;
    signal is_first_frame_reg, is_first_frame_d1_reg, is_first_frame_d2_reg : std_logic := '1';
    signal scaled_sample_reg : t_product_array; -- Stage 3: 48-bit Q32.16 fixed-point DSP multiplier products.
    signal output_a_reg      : t_sample_array;  -- Stage 4: Clamped 16-bit saturated output samples.
    signal output_active     : std_logic;       -- '1' when stage 4 is actively streaming valid frame data.

begin

    ----------------------------------------------------------------------------
    -- Control Interface Unpacking
    -- Decode host control registers into named, typed internal control signals.
    ----------------------------------------------------------------------------
    delay_cycles      <= unsigned(control(0)(31 downto 0));
    scale_factor      <= signed(control(1)(C_SCALE_WIDTH - 1 downto 0));
    frame_limit       <= unsigned(control(2)(31 downto 0));
    pre_leveling_bias <= signed(control(3)(15 downto 0));
    reset_cmd         <= control(4)(0);

    ----------------------------------------------------------------------------
    -- Process: InputRegister
    -- Clock Domain: clk (rising edge)
    -- Description:
    --   Latches incoming parallel samples from platform ports (InputA, InputB)
    --   into internal registers to isolate physical I/O routing delays.
    --   Applies 16-bit pre-leveling subtraction to InputA:
    --     input_a_reg(i) <= signed(InputA(i)) - pre_leveling_bias
    --   to cancel analog input DC offsets before capture and accumulation.
    --   Also registers the final lane (M-1) of InputB to provide seamless
    --   boundary continuity for cross-cycle trigger zero-crossing detection.
    ----------------------------------------------------------------------------
    InputRegister: process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' or reset_cmd = '1' then
                input_a_reg                 <= (others => (others => '0'));
                input_a_d1_reg              <= (others => (others => '0'));
                input_b_reg                 <= (others => (others => '0'));
                input_b_previous_sample_reg <= (others => '0');
            else
                for i in 0 to INPUT_INTERLACING_FACTOR - 1 loop
                    input_a_reg(i) <= signed(InputA(i)) - pre_leveling_bias;
                    input_b_reg(i) <= signed(InputB(i));
                end loop;
                input_a_d1_reg              <= input_a_reg;
                input_b_previous_sample_reg <= input_b_reg(INPUT_INTERLACING_FACTOR - 1);
            end if;
        end if;
    end process;

    ----------------------------------------------------------------------------
    -- Process: TriggerDetect
    -- Sensitivity: input_b_previous_sample_reg, input_b_reg
    -- Description:
    --   Combinational priority search across all M interlaced trigger lanes.
    --   Evaluates each lane for a rising zero crossing:
    --     v_prev <= 0 and current_sample > 0
    --   - For lane 0, v_prev is the boundary sample from the previous clock cycle.
    --   - For lanes 1..M-1, v_prev is the adjacent preceding lane in this group.
    --   Selects the first crossing in lane order (priority encoded) and asserts
    --   trigger_found along with the crossing lane index (trigger_offset).
    ----------------------------------------------------------------------------
    TriggerDetect: process(input_b_previous_sample_reg, input_b_reg)
        variable v_previous_sample : signed(15 downto 0);
        variable v_found           : std_logic;
    begin
        v_found        := '0';
        trigger_offset <= 0;
        for i in 0 to INPUT_INTERLACING_FACTOR - 1 loop
            if i = 0 then
                v_previous_sample := input_b_previous_sample_reg;
            else
                v_previous_sample := input_b_reg(i - 1);
            end if;

            if v_found = '0' and v_previous_sample <= 0 and input_b_reg(i) > 0 then
                v_found        := '1';
                trigger_offset <= i;
            end if;
        end loop;
        trigger_found <= v_found;
    end process;

    ----------------------------------------------------------------------------
    -- Process: CaptureControl
    -- Clock Domain: clk (rising edge)
    -- Description:
    --   Main acquisition FSM governing trigger holdoff delay, sequential RAM
    --   read address generation (0..127), and accumulated frame counting:
    --   - IDLE    : Clears RAM address. When trigger_found = '1', latches trigger
    --               offset. If paused, transitions to READ; if delay is 0, goes
    --               to CAPTURE; otherwise loads delay counter and enters DELAY.
    --   - DELAY   : Decrements delay_counter_reg to 1, then enters CAPTURE.
    --   - CAPTURE : Increments memory_address_reg from 0 to 127. At address 127,
    --               returns to IDLE, increments frame_count_reg, and sets
    --               capture_paused_reg = '1' once frame_limit is reached.
    --   - READ    : Increments memory_address_reg from 0 to 127 to read out the
    --               stored average without further accumulation, then returns to IDLE.
    ----------------------------------------------------------------------------
    CaptureControl: process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' or reset_cmd = '1' then
                state_reg          <= IDLE;
                frame_count_reg    <= (others => '0');
                capture_paused_reg <= '0';
                delay_counter_reg  <= (others => '0');
                memory_address_reg <= (others => '0');
                is_first_frame_reg <= '1';
            else
                case state_reg is
                    when IDLE =>
                        memory_address_reg <= (others => '0');
                        if trigger_found = '1' then
                            active_trigger_offset_reg <= trigger_offset;
                            if capture_paused_reg = '1' then
                                state_reg <= READ;
                            else
                                if frame_count_reg = 0 then
                                    is_first_frame_reg <= '1';
                                else
                                    is_first_frame_reg <= '0';
                                end if;
                                delay_counter_reg <= delay_cycles;
                                if delay_cycles = 0 then
                                    state_reg <= CAPTURE;
                                else
                                    state_reg <= DELAY;
                                end if;
                            end if;
                        end if;

                    when DELAY =>
                        if delay_counter_reg <= 1 then
                            state_reg <= CAPTURE;
                        else
                            delay_counter_reg <= delay_counter_reg - 1;
                        end if;

                    when CAPTURE =>
                        if memory_address_reg = C_FRAME_SIZE - 1 then
                            state_reg <= IDLE;
                            if (frame_count_reg + 1) >= frame_limit then
                                capture_paused_reg <= '1';
                            end if;
                            frame_count_reg <= frame_count_reg + 1;
                        else
                            memory_address_reg <= memory_address_reg + 1;
                        end if;

                    when READ =>
                        if memory_address_reg = C_FRAME_SIZE - 1 then
                            state_reg <= IDLE;
                        else
                            memory_address_reg <= memory_address_reg + 1;
                        end if;
                end case;
            end if;
        end if;
    end process;

    ----------------------------------------------------------------------------
    -- Process: InputAlign (Input Barrel Shifter)
    -- Clock Domain: clk (rising edge)
    -- Description:
    --   Rotates incoming parallel samples so that lane index 0 always contains
    --   the exact sample corresponding to the trigger crossing:
    --   - Lanes i < M - offset are drawn from input_a_d1_reg (previous cycle).
    --   - Lanes i >= M - offset are drawn from input_a_reg (current cycle).
    --   This guarantees that memory address 0, lane 0 in accum_mem is normalized
    --   to the trigger event across all capture runs.
    ----------------------------------------------------------------------------
    InputAlign: process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' or reset_cmd = '1' then
                input_a_d2_reg <= (others => (others => '0'));
            else
                for i in 0 to INPUT_INTERLACING_FACTOR - 1 loop
                    if (i + active_trigger_offset_reg) < INPUT_INTERLACING_FACTOR then
                        input_a_d2_reg(i) <= input_a_d1_reg(i + active_trigger_offset_reg);
                    else
                        input_a_d2_reg(i) <= input_a_reg(i + active_trigger_offset_reg - INPUT_INTERLACING_FACTOR);
                    end if;
                end loop;
            end if;
        end if;
    end process;

    ----------------------------------------------------------------------------
    -- Process: MemoryRead (Pipeline Stage 1: Read Address & Pipeline Advancement)
    -- Clock Domain: clk (rising edge)
    -- Description:
    --   Advances read address pipeline stage:
    --   - memory_address_d1_reg <= memory_address_reg
    --   - state_d1_reg          <= state_reg
    --   - is_first_frame_d1_reg <= is_first_frame_reg
    --   The Block RAM read data synchronously emerges into bram_dob / memory_read_data_reg.
    ----------------------------------------------------------------------------
    MemoryRead: process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' or reset_cmd = '1' then
                memory_address_d1_reg <= (others => '0');
                state_d1_reg          <= IDLE;
                is_first_frame_d1_reg <= '1';
            else
                memory_address_d1_reg <= memory_address_reg;
                state_d1_reg          <= state_reg;
                is_first_frame_d1_reg <= is_first_frame_reg;
            end if;
        end if;
    end process;

    ----------------------------------------------------------------------------
    -- Process: Accumulate (Pipeline Stage 2: Latched Read Data & Write Control)
    -- Clock Domain: clk (rising edge)
    -- Description:
    --   Latches RAM read data into memory_read_data_d1_reg and write address into
    --   memory_address_d2_reg.
    --   Advances pipeline flags: state_d2_reg, is_first_frame_d2_reg.
    ----------------------------------------------------------------------------
    Accumulate: process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' or reset_cmd = '1' then
                memory_read_data_d1_reg <= (others => (others => '0'));
                memory_address_d2_reg   <= (others => '0');
                state_d2_reg            <= IDLE;
                is_first_frame_d2_reg   <= '1';
            else
                memory_read_data_d1_reg <= memory_read_data_reg;
                memory_address_d2_reg   <= memory_address_d1_reg;
                state_d2_reg            <= state_d1_reg;
                is_first_frame_d2_reg   <= is_first_frame_d1_reg;
            end if;
        end if;
    end process;

    ----------------------------------------------------------------------------
    -- Block RAM Array Instantiation (simple_dual_one_clock)
    -- Generates one dedicated BRAM per parallel interlaced channel.
    ----------------------------------------------------------------------------
    bram_addrb <= std_logic_vector(memory_address_reg);
    bram_addra <= std_logic_vector(memory_address_d2_reg);
    bram_wea   <= '1' when (state_d2_reg = CAPTURE) else '0';

    GenBRAM: for k in 0 to INPUT_INTERLACING_FACTOR - 1 generate
        BramInst: simple_dual_one_clock
            generic map(
                ADDR_WIDTH => C_ADDR_WIDTH,
                DATA_WIDTH => C_ACC_WIDTH
            )
            port map(
                clk   => clk,
                ena   => '1',
                enb   => '1',
                wea   => bram_wea,
                addra => bram_addra,
                addrb => bram_addrb,
                dia   => bram_dia(k),
                dob   => bram_dob(k)
            );

        -- Map BRAM read output into internal signed array
        memory_read_data_reg(k) <= signed(bram_dob(k));

        -- Overwrite memory on first frame after reset, accumulate on subsequent frames
        bram_dia(k) <= std_logic_vector(resize(input_a_d2_reg(k), C_ACC_WIDTH)) when (is_first_frame_d2_reg = '1') else
                       std_logic_vector(memory_read_data_d1_reg(k) + resize(input_a_d2_reg(k), C_ACC_WIDTH));
    end generate;

    ----------------------------------------------------------------------------
    -- Process: ScaleOutput (Pipeline Stage 3: Fixed-Point DSP Multiplication)
    -- Clock Domain: clk (rising edge)
    -- Description:
    --   Multiplies the accumulator word (memory_read_data_d1_reg) by the
    --   signed fractional scale_factor (Q0.[C_SCALE_WIDTH] format).
    --   Produces a signed product (Q[C_ACC_WIDTH].[C_SCALE_WIDTH] format) in scaled_sample_reg:
    --     - Bits (C_PROD_WIDTH - 1 downto C_SCALE_WIDTH): signed integer portion.
    --     - Bits (C_SCALE_WIDTH - 1 downto 0)          : fractional sub-sample bits.
    --   Advances pipeline control: state_d3_reg <= state_d2_reg.
    ----------------------------------------------------------------------------
    ScaleOutput: process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' or reset_cmd = '1' then
                scaled_sample_reg <= (others => (others => '0'));
                state_d3_reg      <= IDLE;
            else
                state_d3_reg <= state_d2_reg;

                for i in 0 to OUTPUT_INTERLACING_FACTOR - 1 loop
                    scaled_sample_reg(i) <= memory_read_data_d1_reg(i) * scale_factor;
                end loop;
            end if;
        end if;
    end process;

    ----------------------------------------------------------------------------
    -- Process: SaturateOutput (Pipeline Stage 4: 16-Bit Saturation & Clamping)
    -- Clock Domain: clk (rising edge)
    -- Description:
    --   Truncates fractional bits [C_SCALE_WIDTH - 1 downto 0] and inspects overflow guard bits
    --   from (C_PROD_WIDTH - 1 downto C_ACC_WIDTH - 1) to clamp arithmetic
    --   overflows/underflows to the signed 16-bit output range:
    --   - Positive Overflow  (guard bits > 0)   : Clamp to +32767 (0x7FFF).
    --   - Negative Underflow (guard bits < -1)  : Clamp to -32768 (0x8000).
    --   - In-Range           (otherwise)        : Extract C_OUTPUT_WIDTH bits spanning
    --     from (C_ACC_WIDTH - 1) down to (C_ACC_WIDTH - C_OUTPUT_WIDTH).
    --   Advances pipeline control: state_d4_reg <= state_d3_reg.
    ----------------------------------------------------------------------------
    SaturateOutput: process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' or reset_cmd = '1' then
                output_a_reg <= (others => (others => '0'));
                state_d4_reg <= IDLE;
            else
                state_d4_reg <= state_d3_reg;
                for i in 0 to OUTPUT_INTERLACING_FACTOR - 1 loop
                    if scaled_sample_reg(i)(C_PROD_WIDTH - 1 downto C_ACC_WIDTH - 1) > 0 then
                        output_a_reg(i) <= x"7FFF";
                    elsif scaled_sample_reg(i)(C_PROD_WIDTH - 1 downto C_ACC_WIDTH - 1) < -1 then
                        output_a_reg(i) <= x"8000";
                    else
                        output_a_reg(i) <= scaled_sample_reg(i)(C_ACC_WIDTH - 1 downto C_ACC_WIDTH - C_OUTPUT_WIDTH);
                    end if;
                end loop;
            end if;
        end if;
    end process;

    ----------------------------------------------------------------------------
    -- Output Qualification & Port Driving
    -- Description:
    --   output_active is asserted whenever the output pipeline stage (state_d4_reg)
    --   is in CAPTURE or READ, spanning the valid frame window.
    --   - OutputA: Streams saturated 16-bit average samples when active; 0 when idle.
    --   - OutputB: Drives 0x3FFF frame capture marker when active; 0 when idle.
    --   - OutputC: Unused, driven to zero.
    --   - OutputD: Unused, driven to zero.
    ----------------------------------------------------------------------------
    output_active <= '1' when (state_d4_reg = CAPTURE or state_d4_reg = READ) else '0';

    DriveOutputs: for k in 0 to OUTPUT_INTERLACING_FACTOR - 1 generate
        OutputA(k) <= output_a_reg(k) when output_active = '1' else (others => '0');
        OutputB(k) <= x"3FFF"         when output_active = '1' else x"0000";
        OutputC(k) <= (others => '0');
        OutputD(k) <= (others => '0');
    end generate;

end architecture;