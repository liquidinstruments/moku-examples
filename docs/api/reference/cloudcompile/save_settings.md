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
from moku.instruments import MultiInstrument, CloudCompile
m = MultiInstrument('192.168.###.###', platform_id=2)
cc = m.set_instrument(1, CloudCompile)
# set connections
cc.set_control(2, 156)
# Save the current settings of the instrument
cc.save_settings(filename="instrument_state.mokuconf")
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuMultiInstrument('192.168.###.###', 'platform_id', 2);
cc = m.set_instrument(1, MokuCloudCompile);
% set connections
cc.set_control(2, 156);
% Save the current settings of the instrument
cc.save_settings("instrument_state.mokuconf");
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"filename": "instrument_state.mokuconf"}'\
        http://<ip>/api/slot1/cloudcompile/save_settings
```
