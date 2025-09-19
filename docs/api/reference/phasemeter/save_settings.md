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
from moku.instruments import Phasemeter
# Connect to your Moku
i = Phasemeter('192.168.###.###', force_connect=True)

# Configure Channel 1 to no auto acquire, signal frequency at 1 MHz, bandwidth of 100 Hz.
i.set_pm_loop(1, auto_acquire=False, frequency=1e6, bandwidth='100Hz')

# Save the current settings of the instrument
i.save_settings(filename="instrument_state.mokuconf")
```

</code-block>

<code-block title="MATLAB">

```matlab
% Connect to your Moku
m = MokuPhasemeter('192.168.###.###', force_connect=true);

% Configure Channel 1 to no auto acquire, signal frequency at 1 MHz, bandwidth of 100 Hz.
i.set_pm_loop(1,'auto_acquire',false,'frequency',1e6,'bandwidth','100Hz');

% Save the current settings of the instrument
m.save_settings('instrument_state.mokuconf');
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"filename": "instrument_state.mokuconf"}'\
        http://<ip>/api/phasemeter/save_settings
```

</code-block>

</code-group>
