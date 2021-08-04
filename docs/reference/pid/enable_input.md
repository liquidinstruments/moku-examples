---
additional_doc: null
description: Enable or disable the PID channel input(s).
method: post
name: enable_input
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
  description: Enable/Disable input signal
  name: enabled
  param_range: null
  type: boolean
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: enable_input
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
# Enable Input Signal
i.enable_input(1, True)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuPIDController('192.168.###.###', true);
% Configure the Channel 2 PID Controller using gain characteristics
%   Overall Gain = 6dB
%   I Gain       = 20dB 
m.set_by_gain_and_section(2, 'overall_gain', 6.0, 'prop_gain', 20)
% Enable Input Signal
m.enable_input(1, true)
```
</code-block>
</code-group>