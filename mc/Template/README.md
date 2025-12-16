# VHDL coding template

Provided is a blank template for creating HDL files using Moku Compile.

## Entity Ports for CustomInstrument

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
| ExtTrig <small><br> (Moku:Lab, Moku:Pro, and Moku:Delta) | in | std_logic | - |
||
| OutputA | out | signed | 15 downto 0 |
| OutputB | out | signed | 15 downto 0 |
| OutputC <small><br> (Moku:Go and Moku:Pro) | out | signed | 15 downto 0 |
| OutputD <small><br> (Moku:Pro only) | out | signed | 15 downto 0 |
||
| Control | in | array_of_slv | 0 to 15 |
| Status  | out | array_of_slv | 0 to 15 |
||

## Entity Ports for CustomWrapper (soon to be deprecated)

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
| ExtTrig <small><br> (Moku:Lab, Moku:Pro, and Moku:Delta) | in | std_logic | - |
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
