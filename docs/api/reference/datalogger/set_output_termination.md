---
additional_doc: null
description: Configures the output termination on a given channel.
method: post
name: set_output_termination
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
    - default: null
      description: Waveform termination
      name: termination
      param_range: 50Ohm, HiZ
      type: string
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_output_termination
available_on: 'Moku:Pro, Moku:Lab, Moku:Delta'
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import Datalogger
i = Datalogger('192.168.###.###')
i.set_output_termination(1, "HiZ")
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuDatalogger('192.168.###.###');
m.set_output_termination(1, 'HiZ');
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel":1,"termination":"HiZ"}'\
        http://<ip>/api/datalogger/set_output_termination
```

</code-block>

</code-group>
