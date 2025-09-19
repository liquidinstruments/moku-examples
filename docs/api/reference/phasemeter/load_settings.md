---
additional_doc: null
description: Load a previously saved `.mokuconf` settings file into the instrument. To create a `.mokuconf` file, either use `save_settings` or the desktop app.
method: post
name: load_settings
parameters:
    - default: null
      description: The path to load the `.mokuconf` file to.
      name: filename
      param_range: null
      type: string
      unit: null
summary: load_settings
---

<headers/>

See [save_settings](./save_settings.md) to save the settings

<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import Phasemeter
# Connect to your Moku
i = Phasemeter('192.168.###.###', force_connect=True)

# Load the saved settings of the instrument from a .mokuconf file
i.load_settings(filename="instrument_state.mokuconf")
```

</code-block>

<code-block title="MATLAB">

```matlab
% Connect to your Moku
m = MokuPhasemeter('192.168.###.###', force_connect=true);

% Load the saved settings of the instrument from a .mokuconf file
m.load_settings("instrument_state.mokuconf");
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"filename": "instrument_state.mokuconf"}'\
        http://<ip>/api/phasemeter/load_settings
```

</code-block>

</code-group>
