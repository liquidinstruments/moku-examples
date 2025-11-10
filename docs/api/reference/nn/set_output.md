---
additional_doc: null
description: Set the output range for a given output
method: post
name: set_output
parameters:
    - default: null
      description: Target channel
      name: channel
      type: integer
      unit: null
      param_range:
          mokupro: 1, 2, 3, 4
          mokudelta: 1, 2
          mokulab: 1, 2
          mokugo: 1, 2
    - default: null
      description: Enable or disable the output channel
      name: enabled
      type: boolean
      unit: null
      param_range: true, false
    - default: null
      description: Low level
      name: low_level
      type: number
      unit: V
      param_range: 
          mokugo: -2.5 to 2.475
          mokulab: -1 to 0.99
          mokupro: -1 to 0.99
          mokudelta: -1 to 0.99
    - default: null
      description: High level
      name: high_level
      type: number
      unit: V
      param_range:
          mokugo: -2.5 to 2.475
          mokulab: -1 to 0.99
          mokupro: -1 to 0.99
          mokudelta: -1 to 0.99
    - default: null
      description: Disable all implicit conversions and coercions
      name: strict
      type: boolean
      unit: null
      param_range: true, false
summary: set_output
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import MultiInstrument, NeuralNetwork
m = MultiInstrument('192.168.###.###', platform_id=4)
nn = m.set_instrument(1, NeuralNetwork)
nn.set_output(strict=False, channel=1, enabled=True, low_level=-1, high_level=1)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuMultiInstrument('192.168.###.###', 'platform_id', 4);
nn = m.set_instrument(1, MokuNeuralNetwork);
nn.set_output(channel=1, enabled=True, low_level=-1, high_level=1)
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "enabled": true, "low_level": -1, "high_level": 1}'\
        http://<ip>/api/slot1/neuralnetwork/set_output
```

</code-block>

</code-group>

### Sample response

```json
{
    "enabled": True, 
    "high_level": 1.0, 
    "low_level": -1.0
}
```
