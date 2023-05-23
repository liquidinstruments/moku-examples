---
additional_doc: null
description: Configures the output load on a given channel.
method: post
name: set_output_load
parameters:
- default: null
  description: Target output channel to generate waveform on
  name: channel
  param_range:  1, 2, 3, 4
  type: integer
  unit: null
- default: null
  description: Waveform load
  name: load
  param_range: 50Ohm, 1MOhm
  type: string
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_output_load
available_on: "Moku:Pro, Moku:Lab"
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import FrequencyResponseAnalyzer
i = FrequencyResponseAnalyzer('192.168.###.###')
i.set_output_load(1, "1MOhm")
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuFrequencyResponseAnalyzer('192.168.###.###');
m.set_output_load(1, '1MOhm');
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel":1,"load":"1MOhm"}'\
        http://<ip>/api/fra/set_output_load
```
</code-block>

</code-group>
