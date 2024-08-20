---
additional_doc: null
description: Sets output gain levels
method: post
name: set_gain
parameters:
    - default: null
      description: Main output gain
      name: main
      type: number
      unit: dB
      param_range: null
    - default: null
      description: Auxiliary output gain
      name: aux
      type: number
      unit: dB
      param_range: null
    - default: false
      description: Invert main channel gain
      name: main_invert
      type: boolean
      unit: null
      param_range: null
    - default: false
      description: Invert auxiliary channel gain
      name: aux_invert
      type: boolean
      unit: null
      param_range: null
    - default: 0dB
      description: Main output gain range
      name: main_gain_range
      type: string
      unit: dB
      param_range: 0dB, 14dB
    - default: 0dB
      description: Aux output gain range
      name: aux_gain_range
      type: string
      unit: dB
      param_range: 0dB, 14dB
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_gain
---

<headers/>
<parameters/>

### Examples

<code-group>

<code-block title="Python">

```python
from moku.instruments import LockInAmp
i = LockInAmp('192.168.###.###')
i.set_gain(main=10,aux=10)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLockInAmp('192.168.###.###');
m.set_gain('main', 10, 'aux', 10);
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"main": 10, "aux": 10}'\
        http://<ip>/api/lockinamp/set_gain
```

</code-block>

</code-group>

### Sample response

```json
{
    "aux": 10.0,
    "aux_invert": false,
    "main": 10.0,
    "main_invert": false
}
```
