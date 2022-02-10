---
additional_doc: null
description: Sets trigger source and parameters.
method: post
name: set_trigger
parameters:
- default: Edge
  description: Trigger type
  name: type
  param_range: Edge, Pulse
  type: string
  unit: null
- default: Input1
  description: Trigger channel
  name: source
  param_range: 
    mokugo: Input1, Input2, Output1, Output2, ProbeA, ProbeB
    mokupro: Input1, Input2, Input3, Input4, Output1, Output2, Output3, Output4, ProbeA, ProbeB, ProbeC, ProbeD
  type: string
  unit: null
- default: 0
  description: Trigger level
  name: level
  param_range:
    mokugo: -5 to 5
    mokupro: -20 to 20
  type: number
  unit: V
- default: Auto
  description: Trigger mode
  name: mode
  param_range: Auto, Normal, Single
  type: string
  unit: null
- default: Rising
  description: Which edge to trigger on. In Pulse Width modes this specifies whether
    the pulse is positive (rising) or negative (falling), with the 'both' option being
    invalid
  name: edge
  param_range: Rising, Falling, Both
  type: string
  unit: null
- default: Positive
  description: Trigger pulse polarity (Pulse mode only)
  name: polarity
  param_range: Positive, Negative
  type: string
  unit: null
- default: 0.0001
  description: Width of the trigger pulse (Pulse mode only)
  name: width
  param_range: 
    mokugo: 480e-9 to 10
    mokupro: 24.51e-6 to 10
  type: number
  unit: Seconds
- default: LessThan
  description: Trigger pulse width condition (pulse mode only)
  name: width_condition
  param_range: GreaterThan, LessThan
  type: string
  unit: null
- default: 1
  description: The number of trigger events to wait for before triggering
  name: nth_event
  param_range: 0 to 65535
  type: integer
  unit: null
- default: 0
  description: The duration to hold-off Oscilloscope trigger post trigger event
  name: holdoff
  param_range: 0 to 10
  type: number
  unit: Seconds
- default: true
  description: Configure auto or manual hysteresis for noise rejection.
  name: auto_sensitivity
  param_range: null
  type: boolean
  unit: null
- default: false
  description: Configure the Oscilloscope with a small amount of hysteresis to prevent
    repeated triggering due to noise
  name: noise_reject
  param_range: null
  type: boolean
  unit: null
- default: false
  description: Configure the trigger signal to pass through a low pass filter to smooths
    out the noise
  name: hf_reject
  param_range: null
  type: boolean
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_trigger
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import Oscilloscope
# Configure the instrument
# Trigger on input Channel 1, rising edge, 0V
i.set_trigger(type="Edge", source="Input1", level=0)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuOscilloscope('192.168.###.###', true);
%% Configure the instrument
% Trigger on input Channel 1, rising edge, 0V
m.set_trigger('type',"Edge", 'source',"Input1", 'level', 0)
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"type": "Edge", "source": "Input1", "level": 0}'\
        http://<ip>/api/oscilloscope/generate_waveform
```
</code-block>

</code-group>