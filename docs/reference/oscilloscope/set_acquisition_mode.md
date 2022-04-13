---
additional_doc: null
description: Set acquisition mode
method: post
name: set_acquisition_mode
parameters:
- default: Normal
  description: Acquisition Mode
  name: mode
  param_range: Normal, Precision, PeakDetect
  type: string
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_acquisition_mode
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import Oscilloscope
i = Oscilloscope('192.168.###.###')
# Set instrument to desired state
i.set_acquisition_mode(mode="Precision")
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuOscilloscope('192.168.###.###');
% Set instrument to desired state
m.set_acquisition_mode('mode', 'Precision')
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"mode": "Normal"}'\
        http://<ip>/api/oscilloscope/set_acquisition_mode
```
</code-block>

</code-group>

### Sample response
```json
{
  "mode": "Precision"
}
```