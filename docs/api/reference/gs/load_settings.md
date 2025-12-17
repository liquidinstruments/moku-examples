---
additional_doc: To create a `.mokuconf` file, either use `save_settings` or the desktop app.
description: Load a previously saved `.mokuconf` settings file into the instrument
method: post
name: load_settings
parameters:
    - default: null
      description: The path to load the `.mokuconf` file from.
      name: filename
      param_range: null
      type: string
      unit: null
summary: load_settings
available_on: 'Moku:Delta'
---

<headers/>

See [save_settings](./save_settings.md) to save the settings

<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import MultiInstrument, GigabitStreamer 
m = MultiInstrument('192.168.###.###', platform_id=3)
gs = m.set_instrument(1, GigabitStreamer)
# Load the saved settings of the instrument from a .mokuconf file
i.load_settings(filename="gbs_state.mokuconf")
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuMultiInstrument('192.168.###.###', 3)
gs = m.set_instrument(1, @MokuGigabitStreamer)
gs.load_settings("gbs_state.mokuconf")
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
       -H 'Content-Type: application/json'\
       --data '{"file_name": "gbs_state.mokuconf"}'\
       http://<ip>/<slot>/api/gs/load_settings
```

</code-block>

</code-group>
