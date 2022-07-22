---
additional_doc: null
description: Toggle measurement between Input and Input/Output
method: post
name: measurement_mode
parameters:
- default: InOut
  description: Set the measurement mode.
  name: mode
  param_range: In, InOut, InIn1
  type: string
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: measurement_mode
---





<headers/>
<parameters/>

<code-group>
<code-block title="Python">
```python
from moku.instruments import FrequencyResponseAnalyzer

i = FrequencyResponseAnalyzer('192.168.###.###')
# Measure input signal on channel 1
i.fra_measurement()
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuFrequencyResponseAnalyzer('192.168.###.###');
% Measure input signal on channel 1
m.measurement_mode()
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{}'\
        http://<ip>/api/fra/measurement_mode
```
</code-block>

</code-group>
