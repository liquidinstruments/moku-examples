## Sine Generator (48-bit)
The SineGen_48 IP core generates high-resolution, low-distortion sine waveforms. It accepts a 48-bit frequency step input and produces both 16-bit sine and cosine outputs through a 32-bit output port. This module serves as a sine waveform generator sub-block suitable for a range of advanced applications, including phase-locked loops and simultaneous amplitude and frequency modulation (AM and FM).

The port map for the IP core can be shown as: 
| **Port** | **Direction**  | **Use**                     |
| -------- | -------------- | --------------------------- |
| aclk     | Input          | Clock signal, rising edge   |
| aresetn  | Input          | Active-Low synchronous clear. aresetn must be driven Low for a minimum of two cycles to reset the core.   |
| s_axis_config_tvalid | Input | TVALID for frequency configuration channel.  |
| s_axis_config_tdata [47:0]  | Input          | TDATA for frequency configuration channel. 48-bit width.               |
| m_axis_data_tdata [31:0]        | Output          | TDATA for 32-bit output data channel. The most significant 16 bits are sine and the least significant 16 bits are cosine. [31:16] is sine and [15:0] is cosine.    |
| m_axis_data_tvalid   | Output         | TVALID for the output data channel.                |
| m_axis_phase_tdata [47:0]        | Output          | TDATA for output phase channel.    |
| m_axis_phase_tvalid   | Output         | TVALID for the input phase output channel.    |

Ref: [DDS Compiler](https://docs.amd.com/r/en-US/pg141-dds-compiler)