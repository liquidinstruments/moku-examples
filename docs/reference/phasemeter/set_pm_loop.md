---
additional_doc: null
description: Configures the input impedance, coupling, and range for each channel
method: post
name: set_pm_loop
summary: Sets Phasemeter loop parameters
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
- default: 30e6
  description: Frequency of the input signal
  name: frequency
  param_range: 1e3 to 300e6
  type: number
  unit: Hz
- default: 1kHz
  description: Bandwidth of the phase-locked loop
  name: bandwidth
  param_range: 1Hz, 10Hz, 100Hz, 1kHz, 10kHz, 100kHz, 1MHz
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


### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import Phasemeter
i = Phasemeter('192.168.###.###')
# Configure Channel 1 to no auto acquire, signal frequency at 1 MHz, bandwidth of 100 Hz.
i.set_pm_loop(1, auto_acquire=False, frequency=1e6, bandwidth='100Hz')
```
</code-block>

<code-block title="MATLAB">
```matlab
i = MokuPhasemeter('192.168.###.###');
% Configure Channel 1 to no auto acquire, signal frequency at 1 MHz, bandwidth of 100 Hz.
i.set_pm_loop(1,'auto_acquire',false,'frequency',1e6,'bandwidth','100Hz');
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "auto_acquire": false,"frequency": 1e6, "bandwidth": "100Hz"}'\
        http://<ip>/api/phasemeter/set_pm_loop
```
</code-block>

</code-group>