---
additional_doc: The file name should have a `.mokuconf` extension to be compatible with other tools.
description: Save instrument settings to a file
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
available_on: 'Moku:Delta'
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import MultiInstrument, GigabitStreamer 
m = MultiInstrument('192.168.###.###', platform_id=3)
gs = m.set_instrument(1, GigabitStreamer)
gs.save_settings(filename="gbs_state.mokuconf")
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuMultiInstrument('192.168.###.###', 3)
gs = m.set_instrument(1, @MokuGigabitStreamer)
gs.save_settings("gbs_state.mokuconf")
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
       -H 'Content-Type: application/json'\
       --data '{"filename": "gbs_state.mokuconf"}'\
       http://<ip>/<slot>/api/gs/save_settings
```

</code-block>

</code-group>
