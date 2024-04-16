---
additional_doc: null
description: Configure ParallelBus decoder on a given channel
method: post
name: set_parallel_bus_decoder
parameters:
- default: 
  description: Target channel
  name: channel
  param_range: 
  type: integer
  unit: 
- default: 
  description: Sample mode 
  name: sample_mode
  param_range: Rising, Falling, Both
  type: 
  unit: 
- default: 
  description: Number of data bits
  name: data_width
  param_range: 
  type: integer
  unit: 
- default: 
  description: Clock bit
  name: clock_bit
  param_range: 0 to 15
  type: integer
  unit: 
- default: True
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: 
  type: boolean
  unit: 
summary: set_parallel_bus_decoder
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
# Configure PA decoder
i.set_parallel_bus_decoder(1, sample_mode="Rising", data_width=2, clock_bit=3)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuLogicAnalyzer('192.168.###.###');
patterns=[struct('pin',1,'pattern',[0,1,0,1])];
% Configure PG1 to generate pattern on pin2
m.set_pattern_generator(1, 'patterns', patterns, 'divider', 12);
m.set_pin(2, "PG1");
% Configure PA decoder
m.set_parallel_bus_decoder(1, "Rising", 2, 3)
```
</code-block>

<code-block title="cURL">
```bash
# You should create a JSON file with the data content rather than passing
# arguments on the CLI as the lookup data is necessarily very large
$: cat request.json
{
    "channel":1,
    "sample_mode":"Rising",
    "data_width": 2,
    "clock_bit":3
}
$: curl -H 'Moku-Client-Key: <key>' \
    -H 'Content-Type: application/json' \
    --data @request.json \
    http:// <ip >/api/logicanalyzer/set_parallel_bus_decoder
```
</code-block>

</code-group>

