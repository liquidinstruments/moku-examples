---
additional_doc: null
description: Set the linear combination of ADC input signals for a given PID channel.
method: post
name: set_control_matrix
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
  description: ADC input gain for Channel 1
  name: input_gain1
  param_range: 0 to 20
  type: number
  unit: dB
- default: null
  description: ADC input gain for Channel 2
  name: input_gain2
  param_range: 0 to 20
  type: number
  unit: dB
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_control_matrix
---
<headers/>
<parameters/>

Usage in clients, 

<code-group>
<code-block title="Python">
```python
from moku.instruments import PIDController
i = PIDController('192.168.###.###', force_connect=False)
i.set_control_matrix(1, input_gain1=1, input_gain2=0)
i.set_control_matrix(2, input_gain1=0, input_gain2=1)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuPIDController('192.168.###.###', true);
m.set_control_matrix(1, 1, 0);
m.set_control_matrix(2, 0, 1);
```
</code-block>
</code-group>