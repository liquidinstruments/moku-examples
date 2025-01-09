# VHDL coding template

Provided is a blank template for creating HDL files using the Moku Cloud Compile.

## Entity Ports

| Port | In/Out | Type | Range |
| ----- | ----- | ----- | ----- |
| Clk | in | std_logic | - |
| Reset | in | std_logic | - |
||
| InputA | in | signed | 15 downto 0 |
| InputB | in | signed | 15 downto 0 |
| InputC <small><br> (Moku:Pro only) | in | signed | 15 downto 0 |
| InputD <small><br> (Moku:Pro only) | in | signed | 15 downto 0 |
||
| ExtTrig <small><br> (Moku:Lab and Moku:Pro) | in | std_logic | - |
||
| OutputA | out | signed | 15 downto 0 |
| OutputB | out | signed | 15 downto 0 |
| OutputC <small><br> (Moku:Go and Moku:Pro) | out | signed | 15 downto 0 |
| OutputD <small><br> (Moku:Pro only) | out | signed | 15 downto 0 |
||
| Control0 | in | std_logic_vector | 31 downto 0 |
| Control1 | in | std_logic_vector | 31 downto 0 |
| ... | ... | ... | ... |
| Control15 | in | std_logic_vector | 31 downto 0 |
||
