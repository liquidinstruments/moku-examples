---
additional_doc: null
description: Sets the phase wrap function on the phasemeter
method: post
name: set_phase_wrap
available_on: "mokupro"
parameters:
- default: null
  description: Phase wrap point
  name: value
  param_range: Off, 1pi, 2pi, 4pi
  type: string
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_phase_wrap
---


<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import MokuPhasemeter
i = MokuPhasemeter('192.168.###.###', force_connect=False)
i.set_phase_wrap(value='2pi')
```
</code-block>

<code-block title="MATLAB">
```matlab{8}
i = MokuPhasemeter('192.168.###.###', true);
i.set_phase_wrap('2pi');
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"value": "2pi"}'\
        http://<ip>/api/phasemeter/set_phase_wrap
```
</code-block>

</code-group>