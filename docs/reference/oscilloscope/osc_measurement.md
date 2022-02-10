---
additional_doc: This function is a convenience wrapper around the other functions here,
  allowing you to set up common configurations in a single call.
description: Sets up commonly used configurations in the Oscilloscope, including the
  time base, channel signal source, and trigger.
method: post
name: osc_measurement
parameters:
- default: null
  description: Time from the trigger point to the left of screen
  name: t1
  param_range: null
  type: number
  unit: null
- default: null
  description: Time from the trigger point to the right of screen. (Must be a positive
    number, i.e. after the trigger event)
  name: t2
  param_range: null
  type: number
  unit: null
- default: null
  description: Trigger source
  name: trigger_source
  param_range: 
    mokugo: Input1, Input2, Output1, Output2
    mokupro: Input1, Input2, Input3, Input4, Output1, Output2, Output3, Output4, External  
  type: string
  unit: null
- default: null
  description: Which edge to trigger on. Only edge trigger is used with this function,
    pulse trigger can be enabled using set_trigger()
  name: edge
  param_range: Rising, Falling, Both
  type: string
  unit: null
- default: null
  description: Trigger level
  name: level
  param_range: 
    mokugo: -5 to 5
    mokupro: -20 to 20
  type: number
  unit: V
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: osc_measurement
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import Oscilloscope
i = Oscilloscope('192.168.###.###', force_connect=False)
i.osc_measurement(-5e-6, 5e-6, "Input1", "Rising", 1)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuOscilloscope('192.168.###.###', true);
m.osc_measurement(-5e-6, 5e-6, "Input1", "Rising", 1)
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"t1": -5e-6, "t2": 5e-6, "trigger_source": "Input1", "edge": "Rising", "trigger_level": 1}'\
        http://<ip>/api/oscilloscope/osc_measurement
```
</code-block>

</code-group>