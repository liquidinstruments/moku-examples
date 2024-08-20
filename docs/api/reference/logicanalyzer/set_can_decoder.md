---
additional_doc: null
description: Configure CAN decoder on a given channel
method: post
name: set_can_decoder
parameters:
    - default:
      description: Target channel
      name: channel
      param_range:
      type: integer
      unit:
    - default:
      description: Bit index to receive(Rx) signal
      name: data_bit
      param_range: 0 to 15
      type: integer
      unit:
    - default: 500000
      description: Baud Rate
      name: baud_rate
      param_range:
      type: number
      unit:
    - default: False
      description: Bit order. Defaults to false, making it msb_first
      name: lsb_first
      param_range:
      type: boolean
      unit:
    - default: True
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range:
      type: boolean
      unit:
summary: set_can_decoder
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
# Configure CAN decoder
i.set_can_decoder(1, data_bit=3)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLogicAnalyzer('192.168.###.###');
patterns=[struct('pin',1,'pattern',[0,1,0,1])];
% Configure PG1 to generate pattern on pin2
m.set_pattern_generator(1, 'patterns', patterns, 'divider', 12);
m.set_pin(2, "PG1");
% Configure CAN decoder
m.set_can_decoder(1, 3)
```

</code-block>

<code-block title="cURL">

```bash
# You should create a JSON file with the data content rather than passing
# arguments on the CLI as the lookup data is necessarily very large
$: cat request.json
{
    "channel":1,
    "data_bit":3
}
$: curl -H 'Moku-Client-Key: <key>' \
    -H 'Content-Type: application/json' \
    --data @request.json \
    http:// <ip >/api/logicanalyzer/set_can_decoder
```

</code-block>

</code-group>

### Sample response
