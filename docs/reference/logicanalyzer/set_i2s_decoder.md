---
additional_doc: null
description: Configure decoder on the given channel
method: post
name: set_i2s_decoder
parameters:
- default: undefined
  description: Id of channel to configure decoder on
  name: channel
  param_range: 1 to 2
  type: integer
  unit: null
- default: undefined
  description: Bit index to send/receive data
  name: data_bit
  param_range: 1 to 16
  type: integer
  unit: null
- default: undefined
  description: Bit index to send clock signal
  name: clock_bit
  param_range: 1 to 16
  type: integer
  unit: null
- default: undefined
  description: Bit number to select transmitting channel
  name: word_select
  param_range: 1 to 16
  type: integer
  unit: null
- default: false
  description: Bit order for I2S
  name: lsb_first
  param_range: null
  type: boolean
  unit: null
- default: 1
  description: Number of clock cycle to wait after WS transition before data transmission starts
  name: offset
  param_range: 0 to 1
  type: integer
  unit: null  
- default: 8
  description: Number of data bits
  name: data_width
  param_range: 5 to 9
  type: integer
  unit: null  
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_i2s_decoder
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
i.set_i2s_decoder(1, data_bit=1, clock_bit=2, word_select=3, lsb_first=False,
                  offset=1, data_width=8)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuLogicAnalyzer('192.168.###.###');
patterns=[struct('pin',1,'pattern',[0,1,0,1])];
% Configure PG1 to generate pattern on pin2
m.set_pattern_generator(1, 'patterns', patterns, 'divider', 12);
m.set_pin(2, "PG1");
m.set_i2s_decoder(1, 'data_bit', 1, 'clock_bit', 2, 'word_select', 3,... 
                  'lsb_first', false, 'offset', 1, 'data_width', 8);
```
</code-block>

<code-block title="cURL">
```bash
# If the pattern is longer, consider putting the data in a JSON file
# rather than passing on the command line
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel":1,"data_bit":1, "clock_bit":2,"word_select":3, 
        "lsb_first":false, "offset":1, "data_width":8}'\
        http://<ip>/api/logicanalyzer/set_i2s_decoder
```
</code-block>

</code-group>

### Sample response,

```json
{
  "clock_bit": 2, 
  "data_bit": 1, 
  "data_width": 8, 
  "lsb_first": false, 
  "offset": 1, 
  "protocol": "", 
  "word_select": 3
}
```