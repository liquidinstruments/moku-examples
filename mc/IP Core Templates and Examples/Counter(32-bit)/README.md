## Counter (32-bit)
The Counter_32 IP core offers counter implementations utilizing lookup tables (LUTs) and single DSP slices. It supports up/down counting modes with output widths of 32 bits. The counter increments by one on each clock cycle and can be synchronously cleared by pulling the **SCLR** signal high. 

The port map for the IP core can be shown as: 
| **Port** | **Direction**  | **Use**                     |
| -------- | -------------- | --------------------------- |
| Clk     | Input          | Clock signal, rising edge   |
| SCLR  | Input          | Synchronous Clear: forces the output to a low state when driven high.          |
| UP        | Output          | Controls the count direction on an up/down counter. Counts up when high, down when low.   |
| Q[31:0]   | Output         | Counter output 32-bit wide.                |


Ref: [Binary Counter](https://docs.amd.com/v/u/en-US/pg121-c-counter-binary)