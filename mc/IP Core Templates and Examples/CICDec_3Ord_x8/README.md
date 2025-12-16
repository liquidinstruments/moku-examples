## CIC Decimation (x8)
The CIC_Dec_3Ordx8 module implements a 3rd-order Cascaded Integrator-Comb (CIC) decimation filter with a fixed decimation rate of 8. CIC filters are widely used in multi-rate systems where a high sampling rate needs to be reduced. The architecture relies solely on adders, subtractors, and delays, making it ideal for hardware-efficient downsampling applications such as digital receivers and lock-in amplifiers. 

The module accepts 16-bit signed input samples and produces 25-bit signed output samples. The input interface provides handshake signals (**tvalid**, **tready**) for both input and output channels. The filter updates its output only when valid input is received, and the core is ready. 

The port map for the IP core can be shown as: 
| **Port** | **Direction**  | **Use**                     |
| -------- | -------------- | --------------------------- |
| aclk     | Input          | Clock signal, rising edge   |
| s_axis_data_tdata [15:0]  | Input          | TDATA for the Data Input Channel. Carries the unprocessed sample data.               |
| s_axis_data_tvalid  | Input          | TVALID for the Data Input Channel. Used by the external block to signal that it is able to provide data.                 |
| s_axis_data_tready       | Output          | TREADY for the Data Input Channel. Used by the CIC decimator to signal that it is ready to accept data. |
| m_axis_data_tdata [24:0]        | Output          | TDATA for the Data Output Channel. Carries the processed sample data.   |
| m_axis_data_tvalid   | Output         | TVALID for the Data Output Channel. Asserted by the CIC decimator to signal that it is able to provide sample data.                  |

Configuration details 

| **Parameter** | **Value** |
| ------------- | --------- |
| Filter Type   | Decimation |
| Number of stages | 3 |
| Differential Delay | 1 |
| Rate Supported | 8 |
| Input Data Width | 16-bits |
| Output Data Width | 25-bits |
| Latency | 15 clock cycles |

Ref: [CIC Compiler](https://docs.amd.com/v/u/en-US/pg140-cic-compiler)