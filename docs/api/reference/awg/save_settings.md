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
from moku.instruments import ArbitraryWaveformGenerator
# Connect to your Moku
i = ArbitraryWaveformGenerator('192.168.###.###', force_connect=True)

# As input channels are only used when there is a modulation, we will use burst mode in this example
# Add burst modulation to waveform on channel 1, triggering on Input 1 at 0.1V level for 3 cycles
i.burst_modulate(1, "Input1", "NCycle", burst_cycles=3, trigger_level=0.1)
# Save the current settings of the instrument
i.save_settings(filename="instrument_state.mokuconf")
```

</code-block>

<code-block title="MATLAB">

```matlab
% Connect to your Moku
m = MokuArbitraryWaveformGenerator('192.168.###.###', force_connect=true);

% As input channels are only used when there is a modulation, we will use burst mode in this example
% Add burst modulation to waveform on channel 1, triggering on Input 1 at 0.1V level for 3 cycles
m.burst_modulate(2, "Input1", "NCycle",'burst_cycles',3,'trigger_level',0.1);

% Save the current settings of the instrument
m.save_settings('instrument_state.mokuconf');
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"filename": "instrument_state.mokuconf"}'\
        http://<ip>/api/awg/save_settings
```

</code-block>

</code-group>
