---
title: summary | Neural Network
additional_doc: null
description: Returns a short summary of current instrument state
method: get
name: summary
parameters: []
summary: summary
available_on: 'Moku:Pro'
---

<headers/>
<parameters/>

<code-group>
<code-block title="Python">

```python
from moku.instruments import MultiInstrument, NeuralNetwork
m = MultiInstrument('192.168.###.###', platform_id=4)
nn = m.set_instrument(1, NeuralNetwork)
print(i.summary())
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuMultiInstrument('192.168.###.###', 'platform_id', 4);
nn = m.set_instrument(1, MokuNeuralNetwork);
nn.summary();
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        http://<ip>/api/slot1/neuralnetwork/summary
```

</code-block>

</code-group>

### Sample response

```text
Moku:Pro Neural Network
Aquisition rate: 305 kSa/s
Input A - low level -1.000 V, high level 1.000 V
Input B - low level -1.000 V, high level 1.000 V
Input C - low level -1.000 V, high level 1.000 V
Input D - low level -1.000 V, high level 1.000 V
Network input: 4 samples, parallel mode
Layer 1: 16 neurons, activation function Tanh
Layer 2: 16 neurons, activation function Tanh
Layer 3: 2 neurons, activation function Linear
Output A - low level -1.000 V, high level 1.000 V, output enabled
Output B - low level -1.000 V, high level 1.000 V, output enabled
Internal 10 MHz clock
```
