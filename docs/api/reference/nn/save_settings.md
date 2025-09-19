---
additional_doc: null
description: Save instrument settings to a file. The file name should have a `.mokuconf` extension to be compatible with other tools.
method: post
name: save_settings
parameters:
    - default: null
      description: The path to save the `.mokuconf` file to.
      name: filename
      param_range: null
      type: string
      unit: null
summary: save_settings
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import MultiInstrument, NeuralNetwork
# Connect to your Moku
m = MultiInstrument('192.168.###.###', force_connect=True, platform_id=4)

# Set instruments in the slots
nn = m.set_instrument(1, NeuralNetwork)
# Set the input sample rate of the Neural Network
nn.set_input_sample_rate(300)

# Save the current settings of the instrument
nn.save_settings(filename="instrument_state.mokuconf")
```

</code-block>

<code-block title="MATLAB">

```matlab
% Connect to your Moku
m = MokuMultiInstrument('192.168.###.###', 'platform_id', 4);

%% Configure the instruments
nn = m.set_instrument(1, MokuNeuralNetwork);
% Set the input sample rate of the Neural Network
nn.set_input_sample_rate(300);

% Save the current settings of the instrument
nn.save_settings('instrument_state.mokuconf');
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"filename": "instrument_state.mokuconf"}'\
        http://<ip>/api/slot2/neuralnetwork/save_settings
```

</code-block>

</code-group>
