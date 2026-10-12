// Combined DDC/DUC Instrument
// 16x interlaced design at 312.5 MHz clock (5 GS/s effective)
//
// Modes:
//   DDC (Control3[0]=0): InputA -> Mixer -> 6-section IIR -> OutputA(I), OutputB(Q)
//   DUC (Control3[0]=1): InputA(I)*cos - InputB(Q)*sin -> OutputA
//
// Pipeline Latency:
//   DDC: 19 cycles (3 input + 3 NCO + 2 mixer + 12 IIR output)
//   DUC: 8 cycles (3 input + 3 NCO + 2 mixer + 1 combine)

/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off UNDRIVEN */

module CustomInstrumentInterlaced #(
    parameter INPUT_INTERLACING_FACTOR = 16,
    parameter OUTPUT_INTERLACING_FACTOR = 16
) (
    input wire clk,
    input wire reset,
    input wire [31:0] sync,

    input wire signed [15:0] inputa [0:INPUT_INTERLACING_FACTOR-1],
    input wire signed [15:0] inputb [0:INPUT_INTERLACING_FACTOR-1],
    input wire signed [15:0] inputc [0:INPUT_INTERLACING_FACTOR-1],
    input wire signed [15:0] inputd [0:INPUT_INTERLACING_FACTOR-1],

    input wire exttrig [0:INPUT_INTERLACING_FACTOR-1],

    output wire signed [15:0] outputa [0:OUTPUT_INTERLACING_FACTOR-1],
    output wire signed [15:0] outputb [0:OUTPUT_INTERLACING_FACTOR-1],
    output wire signed [15:0] outputc [0:OUTPUT_INTERLACING_FACTOR-1],
    output wire signed [15:0] outputd [0:OUTPUT_INTERLACING_FACTOR-1],

    input wire [31:0] control [0:15],
    output wire [31:0] status [0:15]
);

    // ========================================================================
    // Parameters and Constants
    // ========================================================================
    localparam NUM_LANES = INPUT_INTERLACING_FACTOR;
    localparam LUT_DEPTH = 4096;
    localparam LUT_ADDR_BITS = 12;
    localparam NUM_IIR_SECTIONS = 6;

    // Phase increment constant: K = round(2^64 / 5e9) = 3,689,348,815
    // phase_inc = (freq_hz * K) >> 32
    localparam [63:0] FREQ_TO_PHASE_K = 64'd3689348815;

    // ========================================================================
    // Type Definitions
    // ========================================================================
    typedef logic signed [15:0] sample_t;
    typedef logic signed [31:0] wide_t;
    typedef logic signed [32:0] wider_t;
    typedef logic [31:0] phase_t;
    typedef logic [15:0] coeff_t;  // Q0.16 unsigned coefficient

    // ========================================================================
    // Control Register Extraction
    // ========================================================================
    wire [31:0] carrier_freq_hz = control[0];
    wire [31:0] cutoff_freq_hz  = control[1];
    wire        mode_select     = control[3][0];  // 0=DDC, 1=DUC

    // ========================================================================
    // IO Boundary Registers
    // ========================================================================
    sample_t inputa_io [0:NUM_LANES-1];
    sample_t inputb_io [0:NUM_LANES-1];

    always_ff @(posedge clk) begin
        if (reset) begin
            for (int i = 0; i < NUM_LANES; i++) begin
                inputa_io[i] <= '0;
                inputb_io[i] <= '0;
            end
        end else begin
            for (int i = 0; i < NUM_LANES; i++) begin
                inputa_io[i] <= inputa[i];
                inputb_io[i] <= inputb[i];
            end
        end
    end

    // ========================================================================
    // Input Pipeline (3 stages: d1, d2, d3)
    // ========================================================================
    sample_t input_d1_a [0:NUM_LANES-1];
    sample_t input_d1_b [0:NUM_LANES-1];
    sample_t input_d2_a [0:NUM_LANES-1];
    sample_t input_d2_b [0:NUM_LANES-1];
    sample_t input_d3_a [0:NUM_LANES-1];
    sample_t input_d3_b [0:NUM_LANES-1];

    always_ff @(posedge clk) begin
        if (reset) begin
            for (int i = 0; i < NUM_LANES; i++) begin
                input_d1_a[i] <= '0;
                input_d1_b[i] <= '0;
                input_d2_a[i] <= '0;
                input_d2_b[i] <= '0;
                input_d3_a[i] <= '0;
                input_d3_b[i] <= '0;
            end
        end else begin
            for (int i = 0; i < NUM_LANES; i++) begin
                input_d1_a[i] <= inputa_io[i];
                input_d1_b[i] <= inputb_io[i];
                input_d2_a[i] <= input_d1_a[i];
                input_d2_b[i] <= input_d1_b[i];
                input_d3_a[i] <= input_d2_a[i];
                input_d3_b[i] <= input_d2_b[i];
            end
        end
    end

    // ========================================================================
    // Phase Increment Computation
    // ========================================================================
    // phase_increment = (carrier_freq_hz * K) >> 32
    // This is computed combinationally and registered for use
    logic [63:0] phase_inc_product;
    phase_t phase_increment_r;

    always_comb begin
        phase_inc_product = carrier_freq_hz * FREQ_TO_PHASE_K;
    end

    always_ff @(posedge clk) begin
        if (reset) begin
            phase_increment_r <= '0;
        end else begin
            phase_increment_r <= phase_inc_product[63:32];
        end
    end

    // ========================================================================
    // NCO Phase Accumulator
    // ========================================================================
    phase_t phase_acc;
    phase_t phase_acc_next;

    // Phase accumulator advances by phase_increment * NUM_LANES per clock
    always_comb begin
        phase_acc_next = phase_acc + (phase_increment_r * NUM_LANES);
    end

    always_ff @(posedge clk) begin
        if (reset) begin
            phase_acc <= '0;
        end else begin
            phase_acc <= phase_acc_next;
        end
    end

    // ========================================================================
    // Per-Lane Phase Computation
    // ========================================================================
    // lane_phase[g] = phase_acc + phase_increment * g
    // Pre-compute increments for timing
    phase_t lane_phase [0:NUM_LANES-1];
    phase_t phase_offset [0:NUM_LANES-1];

    // Register the phase offsets (phase_increment * lane_index)
    always_ff @(posedge clk) begin
        if (reset) begin
            for (int i = 0; i < NUM_LANES; i++) begin
                phase_offset[i] <= '0;
            end
        end else begin
            for (int i = 0; i < NUM_LANES; i++) begin
                phase_offset[i] <= phase_increment_r * i[3:0];
            end
        end
    end

    // Compute per-lane phase combinationally from registered values
    always_comb begin
        for (int i = 0; i < NUM_LANES; i++) begin
            lane_phase[i] = phase_acc + phase_offset[i];
        end
    end

    // ========================================================================
    // NCO Sin/Cos LUT (16 replicated pairs for parallel access)
    // ========================================================================
    // LUT address = phase[31:20] (top 12 bits)
    sample_t sin_val [0:NUM_LANES-1];
    sample_t cos_val [0:NUM_LANES-1];
    sample_t sin_val_r [0:NUM_LANES-1];
    sample_t cos_val_r [0:NUM_LANES-1];

    // Replicated sin/cos LUTs per lane with integrated read logic
    genvar g;
    generate
        for (g = 0; g < NUM_LANES; g = g + 1) begin : NCO_LUT
            (* ram_style = "block" *) sample_t sin_lut [0:LUT_DEPTH-1];
            (* ram_style = "block" *) sample_t cos_lut [0:LUT_DEPTH-1];

            initial begin
                $readmemh("sin_lut.mem", sin_lut);
                $readmemh("cos_lut.mem", cos_lut);
            end

            // LUT address for this lane
            logic [LUT_ADDR_BITS-1:0] addr_comb;
            logic [LUT_ADDR_BITS-1:0] addr_reg;

            // Combinational address from phase
            always_comb begin
                addr_comb = lane_phase[g][31:20];
            end

            // Stage 1: Register LUT address
            always_ff @(posedge clk) begin
                if (reset) begin
                    addr_reg <= '0;
                end else begin
                    addr_reg <= addr_comb;
                end
            end

            // Stage 2: LUT read (BRAM output)
            always_ff @(posedge clk) begin
                if (reset) begin
                    sin_val[g] <= '0;
                    cos_val[g] <= '0;
                end else begin
                    sin_val[g] <= sin_lut[addr_reg];
                    cos_val[g] <= cos_lut[addr_reg];
                end
            end

            // Stage 3: Pipeline register for NCO output
            always_ff @(posedge clk) begin
                if (reset) begin
                    sin_val_r[g] <= '0;
                    cos_val_r[g] <= '0;
                end else begin
                    sin_val_r[g] <= sin_val[g];
                    cos_val_r[g] <= cos_val[g];
                end
            end
        end
    endgenerate

    // ========================================================================
    // External LO Mode Selection
    // ========================================================================
    // When carrier_freq_hz = 0:
    //   DDC: Use InputB as cos, zero sin (external LO mode)
    //   DUC: cos(0)=32767, sin(0)=0 (bypass mode)
    wire use_external_lo = (carrier_freq_hz == 32'd0);

    sample_t nco_cos [0:NUM_LANES-1];
    sample_t nco_sin [0:NUM_LANES-1];

    always_comb begin
        for (int i = 0; i < NUM_LANES; i++) begin
            if (use_external_lo && !mode_select) begin
                // DDC external LO mode: use InputB as cos, zero sin
                nco_cos[i] = input_d3_b[i];
                nco_sin[i] = 16'sd0;
            end else begin
                // Normal NCO operation
                nco_cos[i] = cos_val_r[i];
                nco_sin[i] = sin_val_r[i];
            end
        end
    end

    // ========================================================================
    // Mixer: 16x16 -> 32-bit multiply with Q1.15 normalization
    // ========================================================================
    // DDC: I = InputA * cos, Q = InputA * (-sin)
    // DUC: I_mix = InputA * cos, Q_mix = InputB * sin

    // Stage 1: Register mixer inputs (ensures DSP48 AREG/BREG absorption)
    sample_t mixer_in_i [0:NUM_LANES-1];
    sample_t mixer_in_q [0:NUM_LANES-1];
    sample_t mixer_cos [0:NUM_LANES-1];
    sample_t mixer_sin [0:NUM_LANES-1];

    always_ff @(posedge clk) begin
        if (reset) begin
            for (int i = 0; i < NUM_LANES; i++) begin
                mixer_in_i[i] <= '0;
                mixer_in_q[i] <= '0;
                mixer_cos[i] <= '0;
                mixer_sin[i] <= '0;
            end
        end else begin
            for (int i = 0; i < NUM_LANES; i++) begin
                if (mode_select) begin
                    // DUC mode: I input = InputA, Q input = InputB
                    mixer_in_i[i] <= input_d3_a[i];
                    mixer_in_q[i] <= input_d3_b[i];
                end else begin
                    // DDC mode: Both paths use InputA
                    mixer_in_i[i] <= input_d3_a[i];
                    mixer_in_q[i] <= input_d3_a[i];
                end
                mixer_cos[i] <= nco_cos[i];
                mixer_sin[i] <= nco_sin[i];
            end
        end
    end

    // Stage 2: Multiply (DSP48 MREG)
    wide_t mix_prod_i [0:NUM_LANES-1];
    wide_t mix_prod_q [0:NUM_LANES-1];

    always_ff @(posedge clk) begin
        if (reset) begin
            for (int i = 0; i < NUM_LANES; i++) begin
                mix_prod_i[i] <= '0;
                mix_prod_q[i] <= '0;
            end
        end else begin
            for (int i = 0; i < NUM_LANES; i++) begin
                mix_prod_i[i] <= mixer_in_i[i] * mixer_cos[i];
                mix_prod_q[i] <= mixer_in_q[i] * mixer_sin[i];
            end
        end
    end

    // Stage 3: Scale (>>>15 with rounding) and saturate
    // DDC Q path uses -sin, so negate the product
    sample_t mix_out_i [0:NUM_LANES-1];
    sample_t mix_out_q [0:NUM_LANES-1];

    // Saturation function
    function automatic sample_t saturate_32to16(input wide_t val);
        if (val > 32'sd32767)
            return 16'sd32767;
        else if (val < -32'sd32768)
            return -16'sd32768;
        else
            return val[15:0];
    endfunction

    always_ff @(posedge clk) begin
        if (reset) begin
            for (int i = 0; i < NUM_LANES; i++) begin
                mix_out_i[i] <= '0;
                mix_out_q[i] <= '0;
            end
        end else begin
            for (int i = 0; i < NUM_LANES; i++) begin
                // I path: InputA * cos >>> 15 with rounding
                wide_t i_rounded;
                wide_t q_rounded;
                wide_t q_negated;

                i_rounded = (mix_prod_i[i] + 32'sd16384) >>> 15;
                mix_out_i[i] <= saturate_32to16(i_rounded);

                if (mode_select) begin
                    // DUC mode: Q = InputB * sin >>> 15
                    q_rounded = (mix_prod_q[i] + 32'sd16384) >>> 15;
                    mix_out_q[i] <= saturate_32to16(q_rounded);
                end else begin
                    // DDC mode: Q = InputA * (-sin) >>> 15
                    // Negate the product before scaling
                    q_negated = -mix_prod_q[i];
                    q_rounded = (q_negated + 32'sd16384) >>> 15;
                    mix_out_q[i] <= saturate_32to16(q_rounded);
                end
            end
        end
    end

    // ========================================================================
    // IIR Coefficient Computation
    // ========================================================================
    // alpha_q16 = clamp(round(2*pi * f_cutoff * 65536 / 312,500,000), 0, 65535)
    // Simplify: alpha_q16 = clamp(round(f_cutoff * 0.001319469), 0, 65535)
    // Using integer math: alpha = (f_cutoff * 1319) >> 20 (approximation)
    // More accurate: alpha = (f_cutoff * 2199023) >> 31 (matches 2*pi*65536/312.5e6)

    logic [51:0] alpha_product;
    coeff_t alpha_q16;
    logic [16:0] beta_q16;  // 17-bit to hold 65536

    // Constant: round(2 * pi * 65536 / 312500000 * 2^20) = 1382
    // Verification: For 1 MHz cutoff: (1000000 * 1382) >> 20 = 1318
    // For 10 MHz cutoff: (10000000 * 1382) >> 20 = 13179
    localparam [31:0] ALPHA_CONSTANT = 32'd1382;

    always_comb begin
        alpha_product = cutoff_freq_hz * ALPHA_CONSTANT;
    end

    always_ff @(posedge clk) begin
        if (reset) begin
            alpha_q16 <= '0;
            beta_q16 <= 17'd65536;
        end else begin
            // alpha = (f_cutoff * K) >> 20, clamped to [0, 65535]
            logic [31:0] alpha_raw;
            alpha_raw = alpha_product[51:20];
            if (alpha_raw > 32'd65535)
                alpha_q16 <= 16'd65535;
            else
                alpha_q16 <= alpha_raw[15:0];
            beta_q16 <= 17'd65536 - {1'b0, alpha_q16};
        end
    end

    // ========================================================================
    // IIR Filter: 6-Section First-Order Cascade (per lane, I and Q)
    // ========================================================================
    // Each section: y[n] = alpha*x[n] + beta*y[n-1]
    // Q0.16 * Q1.15 = Q1.31, then >>16 to get Q1.15
    // 2-cycle pipeline per section:
    //   Cycle 1: products = alpha*x, beta*y_prev
    //   Cycle 2: sum = products, shift, saturate

    // IIR state arrays: [lane][section]
    // dont_touch ensures state registers aren't optimized away or merged
    (* dont_touch = "true" *) sample_t iir_state_i [0:NUM_LANES-1][0:NUM_IIR_SECTIONS-1];
    (* dont_touch = "true" *) sample_t iir_state_q [0:NUM_LANES-1][0:NUM_IIR_SECTIONS-1];

    // Pipeline registers for IIR
    // Section inputs
    sample_t iir_in_i [0:NUM_LANES-1][0:NUM_IIR_SECTIONS-1];
    sample_t iir_in_q [0:NUM_LANES-1][0:NUM_IIR_SECTIONS-1];

    // Products (stage 1 output)
    // Alpha product: Q0.17 * Q1.15 = Q1.32 (max 33 bits)
    // Beta product: Q0.18 * Q1.15 = Q1.33 (max 34 bits, beta can be 65536)
    logic signed [33:0] iir_prod_alpha_i [0:NUM_LANES-1][0:NUM_IIR_SECTIONS-1];
    logic signed [33:0] iir_prod_alpha_q [0:NUM_LANES-1][0:NUM_IIR_SECTIONS-1];
    logic signed [33:0] iir_prod_beta_i [0:NUM_LANES-1][0:NUM_IIR_SECTIONS-1];
    logic signed [33:0] iir_prod_beta_q [0:NUM_LANES-1][0:NUM_IIR_SECTIONS-1];

    // Section outputs (stage 2 output)
    sample_t iir_out_i [0:NUM_LANES-1][0:NUM_IIR_SECTIONS-1];
    sample_t iir_out_q [0:NUM_LANES-1][0:NUM_IIR_SECTIONS-1];

    // Coefficient registers (replicated per lane AND per section for reduced fanout)
    // Each register feeds only 2 DSP multiply inputs (I and Q for one lane/section)
    (* dont_touch = "true" *) coeff_t alpha_r [0:NUM_LANES-1][0:NUM_IIR_SECTIONS-1];
    (* dont_touch = "true" *) logic [16:0] beta_r [0:NUM_LANES-1][0:NUM_IIR_SECTIONS-1];

    // Register coefficients for all lanes and sections
    always_ff @(posedge clk) begin
        if (reset) begin
            for (int lane = 0; lane < NUM_LANES; lane++) begin
                for (int s = 0; s < NUM_IIR_SECTIONS; s++) begin
                    alpha_r[lane][s] <= '0;
                    beta_r[lane][s] <= 17'd65536;
                end
            end
        end else begin
            for (int lane = 0; lane < NUM_LANES; lane++) begin
                for (int s = 0; s < NUM_IIR_SECTIONS; s++) begin
                    alpha_r[lane][s] <= alpha_q16;
                    beta_r[lane][s] <= beta_q16;
                end
            end
        end
    end

    // Generate IIR filter chains for all lanes
    generate
        for (g = 0; g < NUM_LANES; g = g + 1) begin : IIR_LANE

            // Section 0 input comes from mixer output
            always_ff @(posedge clk) begin
                if (reset) begin
                    iir_in_i[g][0] <= '0;
                    iir_in_q[g][0] <= '0;
                end else begin
                    iir_in_i[g][0] <= mix_out_i[g];
                    iir_in_q[g][0] <= mix_out_q[g];
                end
            end

            // Generate 6 sections
            for (genvar s = 0; s < NUM_IIR_SECTIONS; s = s + 1) begin : IIR_SECTION

                // Initialize state on reset
                initial begin
                    iir_state_i[g][s] = '0;
                    iir_state_q[g][s] = '0;
                end

                // Stage 1: Compute products
                always_ff @(posedge clk) begin
                    if (reset) begin
                        iir_prod_alpha_i[g][s] <= '0;
                        iir_prod_alpha_q[g][s] <= '0;
                        iir_prod_beta_i[g][s] <= '0;
                        iir_prod_beta_q[g][s] <= '0;
                    end else begin
                        // alpha * x (Q0.16 * Q1.15 = Q1.31)
                        // Use per-lane coefficients to reduce fanout
                        iir_prod_alpha_i[g][s] <= $signed({1'b0, alpha_r[g][s]}) * iir_in_i[g][s];
                        iir_prod_alpha_q[g][s] <= $signed({1'b0, alpha_r[g][s]}) * iir_in_q[g][s];
                        // beta * y_prev (Q0.17 * Q1.15 = Q1.32, but beta <= 65536)
                        iir_prod_beta_i[g][s] <= $signed({1'b0, beta_r[g][s]}) * iir_state_i[g][s];
                        iir_prod_beta_q[g][s] <= $signed({1'b0, beta_r[g][s]}) * iir_state_q[g][s];
                    end
                end

                // Stage 2: Sum, shift (>>16), saturate
                // Optimized for timing: narrower overflow detection, direct saturation
                logic signed [34:0] sum_i_comb, sum_q_comb;
                logic signed [18:0] scaled_i_comb, scaled_q_comb;  // 19 bits after >>16
                sample_t sat_i_comb, sat_q_comb;

                always_comb begin
                    // Sum products with rounding
                    sum_i_comb = iir_prod_alpha_i[g][s] + iir_prod_beta_i[g][s] + 35'sd32768;
                    sum_q_comb = iir_prod_alpha_q[g][s] + iir_prod_beta_q[g][s] + 35'sd32768;

                    // Extract 19-bit scaled result (sum[34:16])
                    scaled_i_comb = sum_i_comb[34:16];
                    scaled_q_comb = sum_q_comb[34:16];

                    // Optimized saturation: check only bits [18:15] for overflow
                    // If [18:15] are all 0s or all 1s, no overflow (value fits in 16-bit signed)
                    // Positive overflow: upper bits are 0001, 0010, 0011, ... 0111 (value > 32767)
                    // Negative overflow: upper bits are 1110, 1101, 1100, ... 1000 (value < -32768)
                    if (scaled_i_comb[18:15] == 4'b0000 || scaled_i_comb[18:15] == 4'b1111)
                        sat_i_comb = scaled_i_comb[15:0];
                    else if (scaled_i_comb[18])  // Negative overflow
                        sat_i_comb = -16'sd32768;
                    else  // Positive overflow
                        sat_i_comb = 16'sd32767;

                    if (scaled_q_comb[18:15] == 4'b0000 || scaled_q_comb[18:15] == 4'b1111)
                        sat_q_comb = scaled_q_comb[15:0];
                    else if (scaled_q_comb[18])  // Negative overflow
                        sat_q_comb = -16'sd32768;
                    else  // Positive overflow
                        sat_q_comb = 16'sd32767;
                end

                always_ff @(posedge clk) begin
                    if (reset) begin
                        iir_out_i[g][s] <= '0;
                        iir_out_q[g][s] <= '0;
                        iir_state_i[g][s] <= '0;
                        iir_state_q[g][s] <= '0;
                    end else begin
                        // Register the saturated values
                        iir_out_i[g][s] <= sat_i_comb;
                        iir_out_q[g][s] <= sat_q_comb;

                        // Update state (feedback) - same value
                        iir_state_i[g][s] <= sat_i_comb;
                        iir_state_q[g][s] <= sat_q_comb;
                    end
                end

                // Connect section output to next section input (except last)
                if (s < NUM_IIR_SECTIONS - 1) begin : CHAIN
                    always_ff @(posedge clk) begin
                        if (reset) begin
                            iir_in_i[g][s+1] <= '0;
                            iir_in_q[g][s+1] <= '0;
                        end else begin
                            iir_in_i[g][s+1] <= iir_out_i[g][s];
                            iir_in_q[g][s+1] <= iir_out_q[g][s];
                        end
                    end
                end

            end : IIR_SECTION
        end : IIR_LANE
    endgenerate

    // Final IIR output (last section output)
    sample_t ddc_out_i [0:NUM_LANES-1];
    sample_t ddc_out_q [0:NUM_LANES-1];

    always_comb begin
        for (int i = 0; i < NUM_LANES; i++) begin
            ddc_out_i[i] = iir_out_i[i][NUM_IIR_SECTIONS-1];
            ddc_out_q[i] = iir_out_q[i][NUM_IIR_SECTIONS-1];
        end
    end

    // ========================================================================
    // DUC Combiner: (I*cos - Q*sin) >> 1
    // ========================================================================
    sample_t duc_out [0:NUM_LANES-1];

    always_ff @(posedge clk) begin
        if (reset) begin
            for (int i = 0; i < NUM_LANES; i++) begin
                duc_out[i] <= '0;
            end
        end else begin
            for (int i = 0; i < NUM_LANES; i++) begin
                // Subtract: I*cos - Q*sin (both already scaled by >>>15)
                // Then >>1 for headroom
                wide_t diff;
                diff = $signed({{16{mix_out_i[i][15]}}, mix_out_i[i]}) -
                       $signed({{16{mix_out_q[i][15]}}, mix_out_q[i]});
                duc_out[i] <= diff[16:1];  // >>1 with truncation
            end
        end
    end

    // ========================================================================
    // Output Multiplexer
    // ========================================================================
    sample_t outa_r [0:NUM_LANES-1];
    sample_t outb_r [0:NUM_LANES-1];

    // Pipeline the mode_select to match output timing
    logic mode_select_d [0:19];  // Delay chain for mode select

    always_ff @(posedge clk) begin
        if (reset) begin
            for (int i = 0; i < 20; i++) begin
                mode_select_d[i] <= 1'b0;
            end
        end else begin
            mode_select_d[0] <= mode_select;
            for (int i = 1; i < 20; i++) begin
                mode_select_d[i] <= mode_select_d[i-1];
            end
        end
    end

    // Use delayed mode select for output mux (8 cycles for DUC path)
    wire mode_at_output = mode_select_d[7];

    always_ff @(posedge clk) begin
        if (reset) begin
            for (int i = 0; i < NUM_LANES; i++) begin
                outa_r[i] <= '0;
                outb_r[i] <= '0;
            end
        end else begin
            for (int i = 0; i < NUM_LANES; i++) begin
                if (mode_at_output) begin
                    // DUC mode: OutputA = DUC result, OutputB = 0
                    outa_r[i] <= duc_out[i];
                    outb_r[i] <= '0;
                end else begin
                    // DDC mode: OutputA = I, OutputB = Q
                    outa_r[i] <= ddc_out_i[i];
                    outb_r[i] <= ddc_out_q[i];
                end
            end
        end
    end

    // ========================================================================
    // Output IO Registers and Assignments
    // ========================================================================
    sample_t outa_io [0:NUM_LANES-1];
    sample_t outb_io [0:NUM_LANES-1];

    always_ff @(posedge clk) begin
        if (reset) begin
            for (int i = 0; i < NUM_LANES; i++) begin
                outa_io[i] <= '0;
                outb_io[i] <= '0;
            end
        end else begin
            for (int i = 0; i < NUM_LANES; i++) begin
                outa_io[i] <= outa_r[i];
                outb_io[i] <= outb_r[i];
            end
        end
    end

    // Output wire assignments
    genvar lane;
    generate
        for (lane = 0; lane < OUTPUT_INTERLACING_FACTOR; lane = lane + 1) begin : OUT_ASSIGN
            assign outputa[lane] = outa_io[lane];
            assign outputb[lane] = outb_io[lane];
            assign outputc[lane] = 16'sd0;
            assign outputd[lane] = 16'sd0;
        end
    endgenerate

    // ========================================================================
    // Status Registers
    // ========================================================================
    assign status[0] = {16'd0, alpha_q16};           // Current IIR alpha_q16
    assign status[1] = 32'd0;                         // Reserved
    assign status[2] = phase_increment_r;             // Current NCO phase increment
    assign status[3] = {31'd0, mode_select};          // Current mode
    assign status[4] = 32'd0;
    assign status[5] = 32'd0;
    assign status[6] = 32'd0;
    assign status[7] = 32'd0;
    assign status[8] = 32'd0;
    assign status[9] = 32'd0;
    assign status[10] = 32'd0;
    assign status[11] = 32'd0;
    assign status[12] = 32'd0;
    assign status[13] = 32'd0;
    assign status[14] = 32'd0;
    assign status[15] = 32'd0;

endmodule
