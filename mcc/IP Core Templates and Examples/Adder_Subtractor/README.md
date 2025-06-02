## Adder Subtractor (16-bit)
The AddSubtract_16 module implements a dynamically configurable arithmetic unit capable of performing either addition or subtraction on two signed 16-bit inputs. 

Both input buses (**A** and **B**) and the output (**S**) are treated as signed 16-bit integers (int16). The arithmetic operation is controlled by the **add** signal, while the **ce** (clock enable) signal enables or disables the clock of the module. When **ce** is held high, the module remains continuously active. 

The port map for the IP core can be shown as: 
| **Port** | **Direction**  | **Use**                     |
| -------- | -------------- | --------------------------- |
| A[15:0]  | Input          | Input A bus                 |
| B[15:0]  | Input          | Input B bus                 |
| clk      | Input          | Clock signal, rising edge   |
| add      | Input          | Controls the operation (1-Addition, 0- Subtraction) |
| ce       | Input          | Active-High Clock enable. Set to constant High  |
| S[15:0]  | Output         | Output Bus                  |


Ref: [Adder Subtractor](https://docs.amd.com/v/u/en-US/pg120-c-addsub)