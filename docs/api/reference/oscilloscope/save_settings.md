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
from moku.instruments import Oscilloscope
# Connect to your Moku
i = Oscilloscope('192.168.###.###', force_connect=True)

# Configure the instrument
# View +- 1 ms i.e. trigger in the centre
i.set_timebase(-1e-3, 1e-3)

# Save the current settings of the instrument
i.save_settings(filename="instrument_state.mokuconf")
```

</code-block>

<code-block title="MATLAB">

```matlab
% Connect to your Moku
m = MokuOscilloscope('192.168.###.###', force_connect=true);

%% Configure the instrument
% View +- 1 ms i.e. trigger in the centre
m.set_timebase(-1e-3, 1e-3);

% Save the current settings of the instrument
m.save_settings('instrument_state.mokuconf');
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"filename": "instrument_state.mokuconf"}'\
        http://<ip>/api/oscilloscope/save_settings
```

</code-block>

</code-group>
