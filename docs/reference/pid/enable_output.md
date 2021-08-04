---
additional_doc: null
description: Enable or disable the PID channel output(s)
method: post
name: enable_output
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
  description: Enable/Disable output signal
  name: signal
  param_range: null
  type: boolean
  unit: null
- default: null
  description: Enable/Disable output
  name: output
  param_range: null
  type: boolean
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: enable_output
---

<headers/>
<parameters/>


Usage in clients, 

<code-group>
<code-block title="Python">
```python
from moku.instruments import PIDController
i = PIDController('192.168.###.###', force_connect=False)
# Configure the Channel 2 PID Controller using gain characteristics
#   Overall Gain = 6dB
#   I Gain       = 20dB 
i.set_by_gain(channel=2, overall_gain=6.0, prop_gain=20)
# Enable the output channels of the PID controller
i.enable_output(1, True, True)
i.enable_output(2, True, True)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuPIDController('192.168.###.###', true);
% Configure the Channel 2 PID Controller using gain characteristics
%   Overall Gain = 6dB
%   I Gain       = 20dB 
m.set_by_gain_and_section(2, 'overall_gain', 6.0, 'prop_gain', 20)
% Enable the output channels of the PID controller
m.enable_output(1, 'signal', True, 'output', True);
m.enable_output(2, 'signal', True, 'output', True);
```
</code-block>
</code-group>