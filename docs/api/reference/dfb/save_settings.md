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
from moku.instruments import DigitalFilterBox
# Connect to your Moku
i = DigitalFilterBox('192.168.###.###', force_connect=True)

# Following configuration produces Chebyshev type 1 IIR filter
i.set_filter(1, "3.906MHz", shape="Lowpass", type="ChebyshevI")
# Set the probes to monitor Filter 1 and Output 2
i.set_monitor(1, "Filter1")
i.set_monitor(2, "Output1")

# Save the current settings of the instrument
i.save_settings(filename="instrument_state.mokuconf")
```

</code-block>

<code-block title="MATLAB">

```matlab
% Connect to your Moku
m = MokuDigitalFilterBox('192.168.###.###', force_connect=true);

% Following configuration produces Chebyshev type 1 IIR filter
m.set_filter(1, '3.906MHz', 'shape', 'Lowpass', 'type', 'ChebyshevI');
% Set the probes to monitor Filter 1 and Output 2
m.set_monitor(1, 'Filter1');
m.set_monitor(2, 'Output1');

% Save the current settings of the instrument
m.save_settings("instrument_state.mokuconf");
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"filename": "instrument_state.mokuconf"}'\
        http://<ip>/api/digitalfilterbox/save_settings
```

</code-block>

</code-group>
