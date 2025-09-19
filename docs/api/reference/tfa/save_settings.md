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
from moku.instruments import TimeFrequencyAnalyzer
# Connect to your Moku
i = TimeFrequencyAnalyzer('192.168.###.###', force_connect=True)

# Configure event detector 1
i.set_event_detector(1, source="Input1", threshold=0.1, edge="Rising")
# Configure event detector 2
i.set_event_detector(2, source="Input2", threshold=0.1, edge="Falling")

# Save the current settings of the instrument
i.save_settings(filename="instrument_state.mokuconf")
```

</code-block>

<code-block title="MATLAB">

```matlab
% Connect to your Moku
m = MokuTimeFrequencyAnalyzer('192.168.###.###', force_connect=true);

% Configure event detector 1
m.set_event_detector(1, 'Input1', 'threshold', '0.1', 'edge', 'Rising');
% Configure event detector 2
m.set_event_detector(2, 'Input2', 'threshold', '0.1', 'edge', 'Falling');

% Save the current settings of the instrument
m.save_settings('instrument_state.mokuconf');
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"filename": "instrument_state.mokuconf"}'\
        http://<ip>/api/tfa/save_settings
```

</code-block>

</code-group>
