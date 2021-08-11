---
additional_doc: null
description: Configures the input impedance, coupling, and range for each channel
method: post
name: set_frontend
summary: set_frontend
available_on: "mokupro"

parameters:
- default: null
  description: Target channel
  name: channel
  param_range: 1, 2, 3, 4
  type: integer
  unit: null
- default: false
  description: Automatically acquire the initial frequency of the input signal
  name: auto_acquire
  param_range: null
  type: boolean
  unit: null
- default: null
  description: Frequency of the input signal
  name: frequency
  param_range: 1e3 to 300e6
  type: number
  unit: Hz
- default: null
  description: Bandwidth of the phase-locked loop
  name: bandwidth
  param_range: 10kHz, 2k5Hz, 600Hz, 150Hz, 40Hz, 10Hz
  type: number
  unit: Hz
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null

---





<headers/>
<parameters/>


Usage in clients, 

<code-group>
<code-block title="Python">
```python
from moku.instruments import MokuPhasemeter
i = MokuPhasemeter('192.168.###.###', force_connect=False)
# Configure Channel 1 to no auto acquire, signal frequency at 1 MHz, bandwidth of 40 Hz.
i.set_pm_loop(1, auto_acquire=false, frequency=1e6, bandwidth='40Hz')
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuPhasemeter('192.168.###.###', true);
% Configure Channel 1 to no auto acquire, signal frequency at 1 MHz, bandwidth of 40 Hz.
m.set_pm_loop(1,'auto_acquire',false,'frequency',1e6,'bandwidth','40Hz');
```
</code-block>
</code-group>