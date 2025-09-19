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
from moku.instruments import LogicAnalyzer
# Connect to your Moku
i = LogicAnalyzer('192.168.###.###', force_connect=True)

# Set the time span to 10 ms
i.set_timebase(-5e-3, 5e-3, roll_mode=False)

# Save the current settings of the instrument
i.save_settings(filename="instrument_state.mokuconf")
```

</code-block>

<code-block title="MATLAB">

```matlab
% Connect to your Moku
m = MokuLogicAnalyzer('192.168.###.###', force_connect=true);

% Set the time span to 10 ms
m.set_timebase(-5e-3, 5e-3, 'roll_mode', false);

% Save the current settings of the instrument
m.save_settings("instrument_state.mokuconf");
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"filename": "instrument_state.mokuconf"}'\
        http://<ip>/api/logicanalyzer/save_settings
```

</code-block>

</code-group>
