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
from moku.instruments import PIDController
# Connect to your Moku
i = PIDController('192.168.###.###', force_connect=True)

# Configure the Channel 1 PID Controller using frequency response
# characteristics
#  P = -10dB
i.set_by_frequency(channel=1, prop_gain=-10)
# Set the probes to monitor Output 1 and Output 2
i.set_monitor(1, 'Output1')
i.set_monitor(2, 'Output2')

# Save the current settings of the instrument
i.save_settings(filename="instrument_state.mokuconf")
```

</code-block>

<code-block title="MATLAB">

```matlab
% Connect to your Moku
m = MokuPIDController('192.168.###.###', force_connect=true);

% Configure the Channel 1 PID Controller using frequency response
% characteristics
%  P = -10dB
m.set_by_frequency(1, 'prop_gain', -20);
% Set the probes to monitor Output 1 and Output 2
m.set_monitor(1, 'Output1');
m.set_monitor(2, 'Output2');

% Save the current settings of the instrument
m.save_settings('instrument_state.mokuconf');
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"filename": "instrument_state.mokuconf"}'\
        http://<ip>/api/pidcontroller/save_settings
```

</code-block>

</code-group>
