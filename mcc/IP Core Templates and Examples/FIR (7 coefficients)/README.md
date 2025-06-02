## FIR Filter (7 Coefficients)

The FIR_Filter_7coef IP core operates at the full clock rate with fixed coefficients, offering a resource-efficient solution for users who need to integrate a low-pass filter into their MCC designs without requiring additional instruments in the MiM setup. The filter coefficients are listed below along with the port definitions.

Filter Coefficients are: 
| **Tap** | **Value** |
| ------- | ----------- |
| 1       | 0.0174062532 |
| 2 | 0.061207387 |
| 3 | 0.166164406 |
| 4 | 0.2552218821 |
| 5 | 0.166164406 |
| 6 | 0.061207387 |
| 7 | 0.01740632 |


The port map for the IP core can be shown as: 
| **Port** | **Direction**  | **Use**                     |
| -------- | -------------- | --------------------------- |
| aclk     | Input          | Clock signal, rising edge   |
| s_axis_data_tvalid  | Input          | TVALID for input data channel. Asserted by external block to indicate data is available for transfer.               |
| s_axis_data_tready      |  Output  | TREADY for input data channel. Asserted by core to indicate core is ready to accept data.   |
| s_axis_data_tdata [15:0]        | Output          | TDATA for input data channel. Conveys the data stream to be filtered.     |
| m_axis_data_tvalid   | Output         | TVALID for output data channel. Asserted by core to indicate data is available for transfer.                |
| m_axis_data_tdata [23:0]        | Output          | TDATA for the output data channel. This is the filtered data stream.  |




>For applications requiring adjustable coefficients, users can either use the Moku FIR Filter Builder or recompile a custom FIR filter IP core.

Ref: [FIR Compiler](https://docs.amd.com/r/en-US/pg149-fir-compiler)