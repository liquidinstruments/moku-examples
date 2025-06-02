## CORDIC Translate (Cartesian to Polar)
The CORDIC core implements the Coordinate Rotation Digital Computer (CORDIC) algorithm, an iterative method used to compute trigonometric functions and, more generally, to solve equations involving hyperbolic and square root operations. This Cordic_Translate_16 core transforms input signals from their Cartesian representation (real and imaginary components) into their corresponding polar form (amplitude and phase). The total computational latency of the module is 20 clock cycles. 

The port map for the IP core can be shown as: 
| **Port** | **Direction**  | **Use**                     |
| -------- | -------------- | --------------------------- |
| aclk     | Input          | Clock signal, rising edge   |
| s_axis_cartesian_tvalid | Input | Handshake signal for channel S_AXIS_CARTESIAN.  |
| s_axis_cartesian_tdata [31:0]  | Input          | Depending on Functional Configuration, this port has one or two subfields - X_IN and Y_IN. X_IN is stored in [15:0] and Y_IN is in [31:16]. These are the Cartesian operands. Each subfield is 16-bit wide. X_IN and Y_IN both have 14 fractional bits and 2 integer bits.          |
| m_axis_dout_tdata [31:0]        | Output          | Depending on Functional Configuration this port contains the following subfields. AMPLITUDE_OUT, PHASE_OUT. AMPLITUDE_OUT is [15:0] and PHASE_OUT is [31:16]. Each subfield is 16-bit wide. AMPLITUDE_OUT has 14 fractional bits and 2 integer bits. PHASE_OUT has 13 fractional bits and 3 integer bits with a unit of radians   |
| m_axis_dout_tvalid   | Output         | Handshake signal for output channel.                   |

Ref: [CORDIC](https://docs.amd.com/v/u/en-US/pg105-cordic)