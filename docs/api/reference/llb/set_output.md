---
additional_doc: null
description: Configures the desired output channel
method: post
name: set_output
parameters:
    - default: null
      description: Target output channel to configure
      name: channel
      param_range:
          mokugo: 1, 2
          mokulab: 1, 2
          mokupro: 1, 2, 3, 4
          mokudelta: 1, 2, 3, 4, 5, 6, 7, 8
      type: integer
      unit: null
    - default: null
      description: Enable/Disable control signal
      name: signal
      param_range: null
      type: boolean
      unit: null
    - default: null
      description: Enable/Disable output signal
      name: output
      param_range: null
      type: boolean
      unit: null
    - default: 0dB
      description: Output gain range
      name: gain
      param_range: 0dB, 14dB
      type: string
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_output
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import LaserLockBox
i = LaserLockBox('192.168.###.###')
i.set_output(1, signal=True, output=True)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLaserLockBox('192.168.###.###');
m.set_output(1, 'signal', true, 'output', true);
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"output": True, "signal": True}'\
        http://<ip>/api/laserlockbox/set_output
```

</code-block>

</code-group>
