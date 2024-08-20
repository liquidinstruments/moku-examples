---
additional_doc: null
description: Configures the LIA low-pass filter
method: post
name: set_filter
parameters:
    - default: null
      description: Filter corner frequency
      name: corner_frequency
      type: number
      unit: Hz
    - default: Slope6dB
      description: Filter slope per octave
      name: slope
      param_range: Slope6dB, Slope12dB, Slope18dB, Slope24dB
      type: string
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_filter
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import LockInAmp
i = LockInAmp('192.168.###.###')
i.set_filter(corner_frequency=100,slope="Slope6dB")
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLockInAmp('192.168.###.###');
m.set_filter('corner_frequency',100,'slope','Slope6dB')
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"corner_frequency":100,"slope":"Slope6dB"}'\
        http://<ip>/api/lockinamp/set_filter
```

</code-block>

</code-group>

### Sample response

```json
{
    "corner_frequency": 99.9774868744271,
    "slope": "Slope6dB"
}
```
