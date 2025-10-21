---
additional_doc: null
description: Configures the output load on a given channel.
method: post
name: set_output_gain
parameters:
    - default: null
      description: Target output channel to generate waveform on
      name: channel
      param_range:
          mokulab: 1, 2
          mokupro: 1, 2, 3, 4
          mokudelta: 1, 2, 3, 4, 5, 6, 7, 8
      type: integer
      unit: null
    - default: 0dB
      description: Output gain
      name: gain
      param_range: 
          mokulab: 
          mokupro: 0dB, 14dB
          mokudelta: 0dB, 20dB
      type: string
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_output_gain
available_on: 'Moku:Delta, Moku:Pro, Moku:Lab'
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import PIDController
i = PIDController('192.168.###.###')
i.set_output_gain(1, "0dB")
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuPIDController('192.168.###.###');
m.set_output_gain(1, '0dB');
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel":1,"load":"0dB"}'\
        http://<ip>/api/pidcontroller/set_output_gain
```

</code-block>

</code-group>
