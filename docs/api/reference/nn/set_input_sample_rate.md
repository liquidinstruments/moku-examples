---
additional_doc: null
description: Set the input sample rate
method: post
name: set_input_sample_rate
parameters:
    - default: null
      description: Input sample rate
      name: sample_rate
      type: number
      unit: Sa/s
      param_range: 0 to 305175.781
    - default: True
      description: Disable all implicit conversions and coercions
      name: strict
      type: boolean
      unit: null
      param_range: true, false
summary: set_input_sample_rate
available_on: 'Moku:Pro'

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
nn.set_input_sample_rate(300)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuMultiInstrument('192.168.###.###', 'platform_id', 4);
nn = m.set_instrument(1, MokuNeuralNetwork);
nn.set_input_sample_rate(300);
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"sample_rate": 300}'\
        http://<ip>/api/slot2/neuralnetwork/set_input_sample_rate
```

</code-block>

</code-group>

### Sample response

```json
{
    "sample_rate": 300.0,
}
```
