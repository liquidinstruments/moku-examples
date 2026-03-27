# Combined DDC/DUC Instrument Specification

## 1. Overview

This instrument implements a dual-mode digital frequency converter on the Moku:Delta platform. It operates as either a Digital Down Converter (DDC) for narrowband I/Q extraction, or a Digital Up Converter (DUC) for wideband RF synthesis. Mode selection is via Control3[0].

The instrument uses 16x interlacing at a 312.5 MHz clock, giving an effective sample rate of 5 GS/s. Both DDC and DUC paths produce 16 distinct output samples per clock cycle at full 5 GS/s effective bandwidth.

```
                                                 DDC path (Control3[0]=0)
                                                ┌──────────────────────────────────────────────┐
                                                │                                              │
InputA ──[d1]──[d2]──[d3]──┬── × lo_cos ── >>15 ── per-lane IIR(6) ──────────────────────────► OutputA (I)
                            │                      (16 independent lanes, 16 distinct samples)
              ┌─────────────┤
              │ DDC: InputA │
              │ DUC: InputB │
              └──── mux ────┴── × (−lo_sin) ── >>15 ── per-lane IIR(6) ──────────────────────► OutputB (Q)
                                                   │    (16 independent lanes, 16 distinct samples)
InputB ──[d1]──[d2]──[d3]──┘                       │
                                                   │  DUC path (Control3[0]=1)
                                 NCO               │ ┌────────────────────────────────────────────┐
carrier_freq ──► phase_acc ──► sin/cos LUT ──────  └─┤  I×cos − Q×sin  ── >>1 ──────────────────► OutputA
                 (32-bit)      (4096-entry)          │  (per-lane subtract)          OutputB = 0
                                                     └────────────────────────────────────────────┘
```

## 2. Inputs

### 2.1 InputA — Primary Signal

16-bit signed, 16 lanes.

- **DDC mode**: Wideband RF signal to be downconverted. The I mixer path multiplies InputA by cos; the Q mixer path multiplies InputA by **−sin** (negative sine). This sign convention ensures that a tone above the carrier maps to a positive baseband frequency.
- **DUC mode**: In-phase (I) baseband component. Multiplied by cos(2πf_c t).

### 2.2 InputB — Secondary Signal

16-bit signed, 16 lanes.

- **DDC mode, Control0 > 0**: Ignored.
- **DDC mode, Control0 = 0**: External LO. Replaces the NCO cosine output in the I mixer path. The sin path is zeroed (no quadrature without a Hilbert transform).
- **DUC mode**: Quadrature (Q) baseband component. Multiplied by sin(2πf_c t).

### 2.3 InputC, InputD

Unused. Not connected internally.

## 3. Outputs

### 3.1 DDC Mode (Control3[0] = 0)

| Port | Content |
|------|---------|
| OutputA | Filtered I (in-phase) channel. Each of the 16 lanes carries a **distinct** time sample at full 5 GS/s effective bandwidth. |
| OutputB | Filtered Q (quadrature) channel. Each of the 16 lanes carries a **distinct** time sample at full 5 GS/s effective bandwidth. |
| OutputC | Zero. |
| OutputD | Zero. |

### 3.2 DUC Mode (Control3[0] = 1)

| Port | Content |
|------|---------|
| OutputA | Up-converted RF signal: InputA × cos(2πf_c t) − InputB × sin(2πf_c t). Each of the 16 lanes carries a distinct time sample at full 5 GS/s bandwidth. |
| OutputB | Zero. |
| OutputC | Zero. |
| OutputD | Zero. |

## 4. Control Registers

All control registers are 32-bit and read dynamically every clock cycle. Default values are the power-on reset state.

| Register               | Bits     | Name               | Mode | Format          | Default | Description                                                          |
|:-----------------------|:--------:|:------------------:|:----:|:----------------|:-------:|:---------------------------------------------------------------------|
| `Control0`             | `[31:0]` | `CarrierFrequency` | Both | Unsigned 32-bit |   `0`   | NCO phase increment. `0` selects external LO (DDC) or bypass (DUC). |
| `Control1`             | `[31:0]` | `CutoffFrequency`  | DDC  | Unsigned 32-bit |   `0`   | IIR lowpass cutoff in Hz. Controls per-lane IIR coefficient.         |
| `Control2`             | `[31:0]` | —                  | —    | —               |   `0`   | Reserved. Write as zero.                                             |
| `Control3`             | `[0]`    | `ModeSelect`       | Both | Single bit      |   `0`   | `0` = DDC, `1` = DUC.                                               |
| `Control3`             | `[31:1]` | —                  | —    | —               |   `0`   | Reserved. Write as zero.                                             |
| `Control4`–`Control15` | `[31:0]` | —                  | —    | —               |   `0`   | Reserved. Write as zero.                                             |

### 4.1 Control0 — Carrier Frequency

**Bits**: [31:0]
**Applies to**: Both modes.
**Range**: 0 to 2,000,000,000 (2 GHz).
**Units**: Hz.

Sets the NCO carrier frequency. The NCO converts this to a 32-bit phase increment:

```
phase_increment = (carrier_freq_hz × 3,689,348,815) >> 32
```

Frequency resolution is 5 GHz / 2^32 = 1.16 Hz.

**When Control0 = 0:**

| Mode | Behaviour |
|------|-----------|
| DDC | External LO mode. InputB replaces the NCO cosine; the sine path is zeroed. OutputB (Q) will be near-zero. |
| DUC | Natural bypass. The NCO produces cos(0) = +32767 and sin(0) = 0. OutputA = InputA × 0.5 (attenuated by the mixer and subtractor scaling stages). InputB has no effect. |

### 4.2 Control1 — Cutoff Frequency

**Bits**: [31:0]
**Applies to**: DDC mode only. Ignored in DUC mode.
**Units**: Hz.

Controls the per-lane IIR lowpass filter cutoff. The IIR consists of 6 cascaded first-order sections per lane: `y[n] = α·x[n] + (1−α)·y[n-1]`. Each of the 16 I lanes and 16 Q lanes has its own independent IIR filter chain (32 chains total, 192 first-order sections).

The coefficient α is computed from Control1:

```
α = 2π × f_cutoff / 312,500,000
```

In Q0.16 fixed-point:

```
alpha_q16 = clamp(round(2π × f_cutoff × 65536 / 312,500,000), 0, 65535)
beta_q16  = 65536 − alpha_q16
```

**Recomputation timing:** α and β are recomputed whenever Control1 changes. Since Control1 is read every clock cycle, the coefficient registers should be updated combinationally from Control1, then registered for use by the IIR.

**Examples:**

| Control1 (Hz) | alpha_q16 | Per-section −3 dB (Hz) |
|---|---|---|
| 10,000 | 13 | 9.87 kHz |
| 100,000 | 132 | 100 kHz |
| 500,000 | 659 | 500 kHz |
| 1,000,000 | 1,318 | 1.00 MHz |
| 5,000,000 | 6,589 | 5.00 MHz |
| 10,000,000 | 13,177 | 10.0 MHz |
| 50,000,000 | 65,535 (clamped) | ~49.7 MHz (passthrough) |

**Minimum useful cutoff:** ~760 Hz per section (alpha_q16 = 1). For the 6-section cascade, the effective −3 dB is ~266 Hz.

**Maximum useful cutoff:** ~49.7 MHz per section (alpha_q16 = 65535). Above this, the filter clamps to passthrough.

When alpha_q16 = 0 (Control1 = 0), the IIR output holds at its last value (β = 1.0, the section becomes a latch).

### 4.3 Control3 — Mode Select

**Bits**: [0]
**Values**: 0 = DDC, 1 = DUC.

Selects the operating mode. May be changed at runtime (see Section 7).

**Bits [31:1]**: Reserved. Write as zero.

### 4.4 Control4–Control15

Reserved. Write as zero.

## 5. Status Registers

| Register | Bits | Content |
|----------|------|---------|
| Status0 | [15:0] | Current IIR alpha_q16 value. |
| Status1 | [31:0] | Reserved (zero). |
| Status2 | [31:0] | Current NCO phase increment (raw 32-bit value). |
| Status3 | [0] | Current mode: 0 = DDC, 1 = DUC. |
| Status4–Status15 | [31:0] | Zero. |

## 6. Operating Modes

### 6.1 DDC — Narrowband Receiver

**Configuration**: Control3[0] = 0, Control0 > 0.

Downconverts a wideband RF input to narrowband baseband I/Q. The NCO generates cos and −sin at the carrier frequency. For each of the 16 lanes independently:

1. **Mixer**: InputA[lane] × cos(phase[lane]) → I; InputA[lane] × (−sin(phase[lane])) → Q. Products are arithmetically right-shifted by 15 bits (Q1.15 normalisation) and saturated to 16 bits.
2. **Per-lane IIR**: The mixer output passes through 6 cascaded first-order IIR lowpass sections. Each section computes `y[n] = α·x[n] + (1−α)·y[n-1]` where α is derived from Control1.
3. **Output**: The filtered values are placed directly on OutputA (I) and OutputB (Q). Each lane carries a distinct time sample.

Signals within the passband (set by Control1) of the carrier frequency appear as low-frequency I/Q components on the output. Signals outside the passband are attenuated by the per-lane IIR (120 dB/decade rolloff from the 6-section cascade).

**Example** — extract a 1 MHz channel centered at 50 MHz:
```
Control0 = 50000000
Control1 = 1000000
Control3 = 0
```

### 6.2 DDC with External LO

**Configuration**: Control3[0] = 0, Control0 = 0.

Same signal path as Section 6.1, except InputB replaces the NCO cosine and the sine path is zeroed. This provides single-channel (I only) downconversion using an external reference. OutputB will be near-zero.

### 6.3 DUC — Wideband Up Converter

**Configuration**: Control3[0] = 1, Control0 > 0.

Up-converts baseband I/Q to RF. InputA (I) is multiplied by cos(2πf_c t), InputB (Q) is multiplied by sin(2πf_c t), and the results are subtracted per-lane:

```
OutputA[lane] = (InputA[lane] × cos − InputB[lane] × sin) >> 16
```

The `>> 16` represents the combined effect of the mixer's `>>> 15` (Q1.15 fixed-point normalisation) and the subtractor's `>> 1` (headroom scaling). The resulting gain is approximately 0.5.

**Upper sideband** — I = cos(ω_m t), Q = sin(ω_m t):
```
OutputA = (A/2) × cos((ω_c + ω_m) t)
```

**Lower sideband** — I = cos(ω_m t), Q = −sin(ω_m t):
```
OutputA = (A/2) × cos((ω_c − ω_m) t)
```

**Double sideband (DSB-SC)** — Q = 0:
```
OutputA = (A/2) × InputA × cos(ω_c t)
```

**Example** — generate upper sideband at 51 MHz from 1 MHz baseband:
```
Control0 = 50000000
Control3 = 1
InputA   = cos(2π × 1 MHz × t) × amplitude
InputB   = sin(2π × 1 MHz × t) × amplitude
```

### 6.4 DUC Bypass

**Configuration**: Control3[0] = 1, Control0 = 0.

Passes InputA to OutputA at 0.5× gain (−6 dB). The NCO produces cos(0) = +32767 and sin(0) = 0, so the Q path contributes nothing. InputB is ignored.

## 7. Mode Switching

Control3[0] may be written at any time.

**DDC to DUC**: The DUC output settles within 7 clock cycles. No overflow risk.

**DUC to DDC**: The per-lane IIR filters will be in an arbitrary state. The IIR settling time is approximately 6/(2π × f_cutoff) seconds. For a 1 MHz cutoff this is ~1.0 µs (~312 clock cycles).

## 8. Pipeline Latency

### 8.1 Shared Stages (both modes)

| Stage | Cycle | Operation |
|-------|-------|-----------|
| 1 | N | Input capture (`input_d1`) |
| 2 | N+1 | Input pipeline (`input_d2`) |
| 3 | N+2 | Input pipeline (`input_d3`); NCO LUT address registered |
| 4 | N+3 | NCO LUT read (BRAM output) |
| 5 | N+4 | NCO pipeline register; input and NCO aligned |
| 6 | N+5 | Mixer multiply (16 × 16 → 32 product) |
| 7 | N+6 | Mixer scale (`>>> 15`, saturate to 16-bit) |

### 8.2 DDC Stages

| Stage | Cycle | Operation |
|-------|-------|-----------|
| 8 | N+7 | IIR section 0: α·x + β·y_prev (per lane, multiply + accumulate) |
| 9 | N+8 | IIR section 0: register output |
| 10 | N+9 | IIR section 1: α·x + β·y_prev (per lane) |
| 11 | N+10 | IIR section 1: register output |
| 12 | N+11 | IIR section 2: α·x + β·y_prev (per lane) |
| 13 | N+12 | IIR section 2: register output |
| 14 | N+13 | IIR section 3: α·x + β·y_prev (per lane) |
| 15 | N+14 | IIR section 3: register output |
| 16 | N+15 | IIR section 4: α·x + β·y_prev (per lane) |
| 17 | N+16 | IIR section 4: register output |
| 18 | N+17 | IIR section 5: α·x + β·y_prev (per lane) |
| 19 | N+18 | IIR section 5: register output → OutputA/OutputB |

**Total DDC latency**: 19 clock cycles from input to output.

### 8.3 DUC Stages

| Stage | Cycle | Operation |
|-------|-------|-----------|
| 8 | N+7 | Per-lane subtraction (I×cos − Q×sin), `>> 1`, output register |

**Total DUC latency**: 7 clock cycles from input to output.

## 9. Signal Gain

### 9.1 DDC

The mixer produces a Q1.15 normalised output. The IIR sections are unity-gain at DC. The overall DDC passband gain is approximately 0.5× (−6 dB) due to the mixer normalisation.

For a sinusoidal input at exactly the carrier frequency (DC in baseband), the I output amplitude is approximately `InputA_amplitude / 2`. The Q output is approximately zero.

### 9.2 DUC

The gain from input to output is 0.5× (−6 dB), comprising:

1. Mixer: 16-bit × 16-bit = 32-bit product, arithmetic right shift by 15 → approximately unity in Q1.15 format.
2. Subtractor: `>> 1` for headroom in the I − Q subtraction → 0.5× additional.

For a DC input of amplitude A on InputA with InputB = 0 and Control0 > 0, the output peak is A/2.

## 10. NCO

The NCO uses a 32-bit phase accumulator with 4096-entry sin/cos BRAM lookup tables (16-bit signed, one pair replicated per lane for parallel access). The LUT is addressed by the top 12 bits of each lane's phase.

Per-lane phase is computed combinationally:

```
lane_phase[g] = phase_acc + phase_increment × g
```

where `phase_acc` advances by `phase_increment × 16` each clock cycle. This avoids a per-lane registered multiply, at the cost of a combinational multiply on the phase path. The combinational multiply is safe at 312.5 MHz because it feeds into a registered LUT address (not a DSP critical path).

## 11. Limitations

1. **DUC bypass gain is 0.5×**, not unity. An explicit bypass pipeline would restore unity gain but adds mux complexity.

2. **Minimum DDC cutoff is ~760 Hz** (single section) / ~266 Hz (6-section cascade). Below this, alpha_q16 quantises to zero and the filter output freezes. For sub-kHz narrowband applications, a wider fixed-point representation (e.g., Q0.24) would be needed.

3. **External LO has no quadrature**. In DDC mode with Control0 = 0, only the cosine path uses InputB. The sine path is zeroed. OutputB (Q) is therefore near-zero in this mode.

4. **Per-lane IIR resource cost**. 192 first-order IIR sections (16 lanes × 2 channels × 6 stages). Resource sharing (time-multiplexing across lanes) can reduce DSP usage at the cost of pipeline complexity.

## 12. Resource Estimate

| Resource | Count | Notes |
|----------|-------|-------|
| BRAM | 32 | 16 lanes × 2 LUTs (sin + cos), 4096 × 16-bit each |
| DSP48 (mixer) | 32 | 16 lanes × 2 mixer multiplies (I cos, Q sin) |
| DSP48 (IIR) | 192 | 16 lanes × 2 (I+Q) × 6 sections × 1 multiply each |
| DSP48 (total) | ~224 | ~5.2% of Moku:Delta's 4,272 DSP48E2 slices |
| Registers (IIR state) | ~6,144 | 192 sections × 32-bit accumulator state |
| LUTs (DUC) | ~800 | 16 subtractors + 16 output muxes |
| Registers (DUC) | ~512 | 16 × 16-bit duc_out registers + mode decode |
