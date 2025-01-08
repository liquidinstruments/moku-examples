---
additional_doc: null
description: Provide a description of the currently loaded network
method: get
name: describe_network
parameters: []
summary: describe_network
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
nn.describe_network()
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuMultiInstrument('192.168.###.###', 'platform_id', 4);
nn = m.set_instrument(1, MokuNeuralNetwork);
nn.describe_network();
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        http://<ip>/api/slot1/neuralnetwork/describe_network
```

</code-block>

</code-group>

### Sample response

```json
{
    "inputs": 4, 
    "layers": [{"activation": "Tanh", "inputs": 4, "outputs": 16}, 
               {"activation": "Tanh", "inputs": 16, "outputs": 16}, 
               {"activation": "Linear", "inputs": 16, "outputs": 2}], 
    "num_input_channels": 4, 
    "num_output_channels": 2, 
    "outputs": 2
}
```
