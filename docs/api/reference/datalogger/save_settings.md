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
from moku.instruments import Datalogger
# Connect to your Moku
i = Datalogger('192.168.###.###', force_connect=True)

# Generate Sine wave on Output1
i.generate_waveform(channel=1, type='Sine', amplitude=1, frequency=10e3)
# Set the Acquisition mode to Precision
i.set_acquisition_mode(mode='Precision')

# Save the current settings of the instrument
i.save_settings(filename="instrument_state.mokuconf")
```

</code-block>

<code-block title="MATLAB">

```matlab
% Connect to your Moku
m = MokuDatalogger('192.168.###.###', force_connect=true);

% Generate Sine wave on Output1
m.generate_waveform(1, 'Sine', 'amplitude',1, 'frequency',10e3);
% Set the Acquisition mode to Precision
m.set_acquisition_mode('mode', 'Precision');

% Save the current settings of the instrument
m.save_settings("instrument_state.mokuconf");
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"filename": "instrument_state.mokuconf"}'\
        http://<ip>/api/datalogger/save_settings
```

</code-block>

</code-group>
