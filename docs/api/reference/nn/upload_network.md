---
additional_doc: null
description: Upload a neural network in .linn JSON format
method: post
name: upload_network
parameters:
    - default: null
      description: A dict like linn data or absolute path to linn file
      name: linn
      type: dict or path
      unit: null
      param_range: null
summary: upload_network
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
nn.upload_network("/path/to/network.linn")
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuMultiInstrument('192.168.###.###', 'platform_id', 4);
nn = m.set_instrument(1, MokuNeuralNetwork);
nn.upload_network("/path/to/network.linn")
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data @path/to/network.linn\
        http://<ip>/api/slot1/neuralnetwork/upload_network
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
