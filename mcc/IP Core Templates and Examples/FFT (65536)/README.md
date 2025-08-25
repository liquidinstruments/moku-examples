## Fast Fourier Transform (1024-points)

FFT_65536 is a variant of FFT_1024, generated using the same IP core compiler but configured with a larger number of FFT points. Although both cores share the same foundation, FFT_65536 has distinct configuration parameters and output data formats to accommodate the increased point count. Detailed port specifications and updated signal connections are provided in Table 11, with changes highlighted. 

Due to hardware resource limitations, the FFT_65536 IP core is not supported on Moku:Go or Moku:Lab and must be deployed on Moku:Pro or Moku:Delta.
The port map for the IP core can be shown as: 
| **Port** | **Direction**  | **Use**                     |
| -------- | -------------- | --------------------------- |
| aclk     | Input          | Clock signal, rising edge   |
| aresetn  | Input          | Active-Low synchronous clear (optional, always take priority over aclken). A minimum aresetn active pulse of two cycles is required.    |
| s_axis_config_tdata[39:0] | Input | TDATA for the Configuration channel. The 0th bit controls the forward or inverse FFT. And [32:1] is the scale of the output. Other bits are empty.   |
| s_axis_config_tvalid  | Input          | TVALID for the Configuration channel. Asserted by the external block to signal that it is able to provide data.               |
| s_axis_config_tready      |  Output  | TREADY for the Configuration channel. Asserted by the core to signal that it is ready to accept configuration data.   |
| s_axis_data_tdata[31:0] | Input | TDATA for the Data Input channel. Carries the unprocessed sample data: real part is [15:0] and imaginary part is [31:16].  |
| s_axis_data_tvalid  | Input          | TVALID for the Data Input channel. Used by the external block to signal that it is able to provide data. It can be set as constant high.               |
| s_axis_data_tready      |  Output  | TREADY for the Data Input channel. Used by the core to signal that it is ready to accept data.   |
| s_axis_data_tlast | Input | TLAST for the Data Input channel. Asserted by the external block on the last sample of the frame. This is not used by the core except to generate the events event_tlast_unexpected and event_tlast_missing events   |
| m_axis_data_tdata [31:0]        | Output          | TDATA for the Data Output channel. Carries the processed sample data real [15:0] and imaginary [31:16]. The signal format is signed 16-bit with 15 fractional bits.  |
| m_axis_data_tuser [15:0]        | Output          | TUSER for the Data Output channel. Carries the index of per-sample information.  |
| m_axis_data_tvalid   | Output         | TVALID for the Data Output channel. Asserted by the core to signal that it is able to provide sample data.                |
| m_axis_data_tready   | Input         | TREADY for the Data Output channel. Asserted by the external slave to signal that it is ready to accept data.                |
| m_axis_data_tlast        | Output          | TLAST for the Data Output channel. Asserted by the core on the last sample of the frame.  |
| event_frame_started | Output | Asserted when the core starts to process a new frame. |
| event_tlast_unexpected | Output | Asserted when the core sees s_axis_data_tlast High on a data sample that is not the last one in a frame.  |
| event_tlast_missing | Output | Asserted when s_axis_data_tlast is Low on the last data sample of a frame. |
| event_status_channel_halt | Output | Asserted when the core tries to write data to the Status channel and it is unable to do so. |
| event_data_in_channel_hal | Output | Asserted when the core requests data from the Data Input channel and none is available. |
| event_data_out_channel_halt | Output | Asserted when the core tries to write data to the Data Output channel and it is unable to do so |


Ref: [XFFT](https://docs.amd.com/r/en-US/pg109-xfft/Port-Descriptions)