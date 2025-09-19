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
from moku.instruments import FrequencyResponseAnalyzer
# Connect to your Moku
i = FrequencyResponseAnalyzer('192.168.###.###', force_connect=True)

# Set output sweep configuration
# 10MHz - 100Hz, 512 sweep points
# Logarithmic sweep ON
# 1msec averaging time, 1msec settling time
# 1 averaging cycle, 1 settling cycle
i.set_sweep(start_frequency=10e6, stop_frequency=100,
      num_points=512, averaging_time=10e-3,
      settling_time=10e-3, averaging_cycles=1,
      settling_cycles=1)

# Save the current settings of the instrument
i.save_settings(filename="instrument_state.mokuconf")
```

</code-block>

<code-block title="MATLAB">

```matlab
% Connect to your Moku
m = MokuFrequencyResponseAnalyzer('192.168.###.###', force_connect=true);

% Set output sweep configuration
% 10MHz - 100Hz, 512 sweep points
% Logarithmic sweep ON
% 1msec averaging time, 1msec settling time
% 1 averaging cycle, 1 settling cycle
m.set_sweep('start_frequency', 10e6, 'stop_frequency', 100, 'num_points', 512, ...
    'averaging_time', 10e-3, 'averaging_cycles', 1,...
    'settling_time', 10e-3, 'settling_cycles', 1);

% Save the current settings of the instrument
m.save_settings("instrument_state.mokuconf");
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"filename": "instrument_state.mokuconf"}'\
        http://<ip>/api/fra/save_settings
```

</code-block>

</code-group>
