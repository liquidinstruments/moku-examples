---
additional_doc: The Arbitrary Waveform Generator has the ability to insert a deadtime between cycles of  the look-up table. This time is specified in cycles of the waveform. During this time, the output will be held at the given dead_voltage. This allows the user to, for example, generate infrequent pulses without using space in the LUT to specify the time between, keeping the full LUT size to provide a high-resolution pulse shape.
description: Configures pulse modulation mode of a channel. 
method: post
name: pulse_modulate
parameters:
- default: null
  description: Target channel
  name: channel
  param_range:
   mokugo: 1, 2
   mokupro: 1, 2, 3, 4
  type: integer
  unit: null
- default: 0
  description: Number of cycles which show the dead voltage.
  name: dead_cycles
  param_range: 1 to 262144
  type: number
  unit: null
- default: 0
  description: Signal level during dead time (the voltage cannot be below low level or above high level)
  name: dead_voltage
  param_range: -5 to 5
  type: number
  unit: V
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: pulse_modulate
---

<headers/>
<parameters/>


### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import ArbitraryWaveformGenerator
i = ArbitraryWaveformGenerator('192.168.###.###', force_connect=False)
# Configure the output waveform in each channel
# Configure modulation on required channels
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuArbitraryWaveformGenerator('192.168.###.###', true);
% Configure the output waveform in each channel

%% Set channel 2 to pulse mode
m.pulse_modulate(1,'dead_cycles',2,'dead_voltage',0);
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel":1, "dead_cycles": 2, "dead_voltage": 0}'\
        http://<ip>/api/awg/pulse_modulate
```
</code-block>

</code-group>