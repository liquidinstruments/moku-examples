---
additional_doc: null
description: Set the voltage range for a given input
method: post
name: set_input
parameters:
    - default: null
      description: Target channel
      name: channel
      type: integer
      unit: null
      param_range:
          mokugo: 1, 2
          mokulab: 1, 2  
          mokupro: 1, 2, 3, 4
          mokudelta: 1, 2
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
          mokugo: -2.475 to 2.5 
          mokulab: -0.99 to 1
          mokupro: -0.99 to 1
          mokudelta: -0.99 to 1
    - default: true
      description: Disable all implicit conversions and coercions
      name: strict
      type: boolean
      unit: null
      param_range: true, false
summary: set_input

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
nn.set_input(channel=1, low_level=-1, high_level=1)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuMultiInstrument('192.168.###.###', 'platform_id', 4);
nn = m.set_instrument(1, MokuNeuralNetwork);
nn.set_inputs(1, -1, 1);
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "low_level": -1, "high_level": 1}'\
        http://<ip>/api/slot1/neuralnetwork/set_input
```

</code-block>

</code-group>

### Sample response

```json
{
    "high_level": 1.0,
    "low_level": -1.0,
}
```
