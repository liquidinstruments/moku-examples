# Full Rate Boxcar Averager (Interlaced)

## Overview

This project implements a high-performance Boxcar Averager designed for the Moku platform. It achieves full-rate signal processing by enabling **multi-sample processing per clock cycle** (interlacing). This architecture allows the instrument to process every sample from the ADC without decimation, even when the FPGA fabric clock runs slower than the ADC sampling rate.

> **Important**: This full-rate implementation requires the **Custom Instrument+** option on the Moku platform and is subject to **export control**.

## Architecture

### Interlacing & Generics

The core feature of this design is its ability to handle multiple samples in parallel.

* **Sample Per Frame**: The number of samples processed per clock cycle is determined by the generic `G_NUM_SAMPLES`.
* **Backend Configuration**: This generic is assigned by the `input_interlacing_factor` constant, which is defined on the backend of the Moku Compile wrapper. This ensures the logic automatically scales to match the specific hardware platform's data path width (e.g., 4 or 16 samples per clock).

## Inputs and Outputs

| Port | Description |
| :--- | :--- |
| **Input A** | Primary signal input (source). |
| **Input B** | Trigger input signal. |
| **Output A** | Main output, selectable between the raw synchronized input or the processed (averaged/gain) result. |
| **Output B** | Auxiliary output, selectable between the boxcar window envelope (for debugging) or the masked sample stream. |

## Control Registers

The instrument behavior is configured via the standard MCC Control registers:

| Register | Name | Description |
| :--- | :--- | :--- |
| `Control(0)` | **Threshold** | Trigger threshold level (Signed 16-bit). |
| `Control(1)` | **Boxcar Delay** | Delay from the trigger event to the start of the integration window in samples (not clock cycles) (Unsigned). |
| `Control(2)` | **Boxcar Length** | Duration of the integration window in samples (not clock cycles) (Unsigned). |
| `Control(3)` | **Num Pulses** | Number of trigger events (pulses) to accumulate before resetting/updating (Unsigned). |
| `Control(4)` | **Gain** | Linear gain applied to the accumulated sum (Signed, 16 integer bits, 16 fractional bits). |
| `Control(5)` | **Output Select** | Configures the output multiplexers:<br>• **Bit 0**: Output A Source (`0` = synced sample, `1` = scaled output)<br>• **Bit 1**: Output B Source (`0` = boxcar window, `1` = boxcar masked samples) |

## Platform Timing Specifications

| Platform | Slot Mode | Sampling Rate | Timing Resolution | Max Trigger Delay / Boxcar Length |
| :--- | :--- | :--- | :--- | :--- |
| **Moku:Go** | 2-slot<br>3-slot | 125 MSa/s<br>62.5 MSa/s | 8 ns<br>16 ns | ~34.36 s<br>~68.72 s |
| **Moku:Lab** | 2-slot<br>3-slot | 500 MSa/s<br>250 MSa/s | 2 ns<br>4 ns | ~8.59 s<br>~17.18 s |
| **Moku:Pro** | 4-slot | 1.25 GSa/s | 800 ps | ~3.44 s |
| **Moku:Delta** | 3-slot<br>5-slot<br>8-slot | 5 GSa/s<br>1.25 GSa/s<br>625 MSa/s | 200 ps<br>800 ps<br>1.6 ns | ~0.86 s<br>~3.44 s<br>~6.87 s |

## Limitations

### Trigger Rate

The current implementation cannot accept two triggers within the same sample frame. Consequently, there is a minimum timing distance required between trigger events, which limits the maximum trigger rate.

* **Constraint**: At least 2 FPGA clock cycles between triggers.

| Platform | FPGA Clock Rate | Minimum Trigger Distance | Max Trigger Rate |
| :--- | :--- | :--- | :--- |
| **Moku:Go** | 31.25 MHz | 64 ns | 15.625 MHz |
| **Moku:Lab** | 125 MHz | 16 ns | 62.5 MHz |
| **Moku:Pro** | 312.5 MHz | 6.4 ns | 156.25 MHz |
| **Moku:Delta** | 312.5 MHz | 6.4 ns | 156.25 MHz |
