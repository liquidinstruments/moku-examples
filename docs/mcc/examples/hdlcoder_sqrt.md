# Simulink™ `sqrt` of signals

Liquid Instruments Moku Cloud Compile works well with High-Level Synthesis tools such as [HDL Coder from Mathworks](https://au.mathworks.com/products/hdl-coder.html). HDL Coder can convert MATLAB and Simulink designs in to VHDL source code, suitable for use in Moku Cloud Compile.

Attention must be paid to data types and widths, as demonstrated in this example where the `sqrt` of the input signals is taken and scaled by a constant before being output. For details on how to use HDL Coder with Moku Cloud Compile, refer to the [Getting Started Guide](https://download.liquidinstruments.com/documentation/app-note/HDLCoderTutorialPart1-MATLAB.pdf).

The Simulink file for this Square Root example can be found in [Gitlab](hhttps://gitlab.com/liquidinstruments/cloud-compile/examples/-/tree/main/hdlcoder_sqrt) and requires MATLAB, Simulink and HDL Coder licenses.

![Simulink Block Diagram of the Square Root example](@mcc/hdlcoder_sqrt/sqrt.png)
