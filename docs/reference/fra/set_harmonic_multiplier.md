---
additional_doc: null
description: set the harmonic multiplier to demodulate at integer harmonic of output
method: post
name: set_harmonic_multiplier
parameters:
- default: 1
  description: Multiplier applied to the fundamental frequency
  name: multiplier
  param_range: 1 to 15
  type: integer
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_harmonic_multiplier
---





<headers/>
<parameters/>

<code-group>
<code-block title="Python">
```python
from moku.instruments import FrequencyResponseAnalyzer

i = FrequencyResponseAnalyzer('192.168.###.###', force_connect=False)
# Measure input signal on channel 1
i.fra_measurement(1, input_only=True, start_frequency=100,
                  stop_frequency=20e6, averaging_cycles=1)
i.set_harmonic_multiplier(multiplier=2)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuFrequencyResponseAnalyzer('192.168.###.###', true);
% Measure input signal on channel 1
i.fra_measurement(1, 'input_only', true, 'start_frequency', 100,
                  'stop_frequency', 20e6, 'averaging_cycles', 1)
m.set_harmonic_multiplier('multiplier', 2)                
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"multiplier": 2}'\
        http://<ip>/api/fra/set_harmonic_multiplier
```
</code-block>

</code-group>

