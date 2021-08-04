---
additional_doc: null
description: Configures the output sweep amplitude and offset. 
method: post
name: set_output
parameters:
- default: null
  description: Target channel
  name: channel
  param_range:
   mokugo: 1, 2
   mokupro: 1, 2, 3, 4
  type: integer
  unit: null
- default: null
  description: Waveform peak-to-peak amplitude
  name: amplitude
  param_range: -5 to 5
  type: number
  unit: V
- default: 0
  description: DC offset applied to the waveform
  name: offset
  param_range: -5 to 5
  type: number
  unit: V
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_output
---





<headers/>

:::tip Note
Ensure the input signal passing through the device under test will not exceed the input range as configured by set_frontend
:::
<parameters/>

<code-group>
<code-block title="Python">
```python
from moku.instruments import FrequencyResponseAnalyzer

i = FrequencyResponseAnalyzer('192.168.###.###', force_connect=False)
# Measure input signal on channel 1
i.fra_measurement(1, input_only=True, start_frequency=100,
                  stop_frequency=20e6, averaging_cycles=1)
# Set output sweep amplitudes and offsets
i.set_output(1, 0.5)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuFrequencyResponseAnalyzer('192.168.###.###', true);
% Measure input signal on channel 1
i.fra_measurement(1, 'input_only', true, 'start_frequency', 100,
                  'stop_frequency', 20e6, 'averaging_cycles', 1)
% Set output sweep amplitudes and offsets 
m.set_output(2, 1)            
```
</code-block>
</code-group>