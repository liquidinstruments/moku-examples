---
additional_doc: null
description: Configure decoder on the given channel
method: post
name: set_uart_decoder
parameters:
- default: null
  description: Id of channel to configure decoder on
  name: channel
  param_range: 1 to 2
  type: integer
  unit: null
- default: null
  description: Bit number to configure as a data pin
  name: data_bit
  param_range: 1 to 16
  type: integer
  unit: null
- default: undefined
  description: Bit order for UART
  name: lsb_first
  param_range: null
  type: boolean
  unit: null
- default: 8
  description: Number of data bits. Cannot be more than 8 if parity bit is enabled
  name: data_width
  param_range: 5 to 9
  type: integer
  unit: null
- default: 1
  description: Number of stop bits.
  name: uart_stop_width
  param_range: 1 to 2
  type: integer
  unit: null
- default: None
  description: Parity for UART.
  name: uart_parity
  param_range: None, Even, Odd
  type: string
  unit: null
- default: 9600
  description: UART baud rate.
  name: uart_baud_rate
  param_range: 1 to 2e6
  type: integer
  unit: baud
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_uart_decoder
---

<headers/>
<parameters/>

### Examples


<code-group>
<code-block title="Python">
```python
from moku.instruments import LogicAnalyzer
i = LogicAnalyzer('192.168.###.###')
patterns = [{"pin": 2, "pattern": [0, 1, 0, 1]}]
# Configure PG1 to generate pattern on pin2
i.set_pattern_generator(1, patterns=patterns, divider=12)
i.set_pin(2, "PG1")
i.set_uart_decoder(1, data_bit=1, lsb_first=False,data_width=8,uart_stop_width=1,
                   uart_parity='None',uart_baud_rate=9600)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuLogicAnalyzer('192.168.###.###');
patterns=[struct('pin',1,'pattern',[0,1,0,1])];
% Configure PG1 to generate pattern on pin2
m.set_pattern_generator(1, 'patterns', patterns, 'divider', 12);
m.set_pin(2, "PG1");
m.set_uart_decoder(1, 'data_bit', 2, 'lsb_first', false, 'data_width', 8,...
                  'uart_stop_width', 1, 'uart_parity','None','uart_baud_rate',9600);
```
</code-block>

<code-block title="cURL">
```bash
# If the pattern is longer, consider putting the data in a JSON file
# rather than passing on the command line
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel":1,"data_bit":2, "lsb_first":false, "data_width":8,
                  "uart_stop_width":1, "uart_parity":"None", "uart_baud_rate":9600}'\
        http://<ip>/api/logicanalyzer/set_uart_decoder
```
</code-block>

</code-group>

### Sample response,

```json
{
  "data_bit": 1,
  "data_width": 8, 
  "lsb_first": false, 
  "protocol": "UART", 
  "uart_baud_rate": 9600, 
  "uart_parity": "None", 
  "uart_stop_width": 1
}
```