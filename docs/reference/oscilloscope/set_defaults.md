---
additional_doc: null
description: Reset the Oscilloscope to default state
method: post
name: set_defaults
parameters: []
summary: set_defaults
---

<headers/>

Default state implies,

- Enable both input channels
- Set input coupling to DC
- Set acquisition mode to Precision
- Set interpolation to SinX
- Set hysterisis mode to Absolute
- Set absolute hysteresis to 10e3-3
- Disable both output channels



::: tip INFO
Reference to any instrument object will always be in default state.
:::



<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import Oscilloscope
i = Oscilloscope('192.168.###.###', force_connect=False)
# Oscilloscope reference i is in default state
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuOscilloscope('192.168.###.###', true);
% Oscilloscope reference m is in default state
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{}'\
        http://<ip>/api/oscilloscope/set_defaults
```
</code-block>

</code-group>