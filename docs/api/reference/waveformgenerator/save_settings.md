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
from moku.instruments import WaveformGenerator
# Connect to your Moku
i = WaveformGenerator('192.168.###.###', force_connect=True)

# Generate a sine wave on channel 1, 0.5 Vpp, 5 kHz
i.generate_waveform(channel=1, type='Sine', amplitude=0.5, frequency=5e3)
# Generate a square wave on channel 2, 1 Vpp, 1 kHz, 50% duty cycle
i.generate_waveform(channel=2, type='Square', amplitude=1.0, frequency=1e3, duty=50)

# Save the current settings of the instrument
i.save_settings(filename="instrument_state.mokuconf")
```

</code-block>

<code-block title="MATLAB">

```matlab
% Connect to your Moku
m = MokuWaveformGenerator('192.168.###.###', force_connect=true);

% Generate a sine wave on Channel 1
% 1Vpp, 10kHz, 0V offset
m.generate_waveform(1, 'Sine','amplitude', 1, 'frequency',1000,'offset',0.2);
% Generate a square wave on Channel 2
% 1Vpp, 10kHz, 0V offset, 50% duty cycle
m.generate_waveform(2, 'Square', 'amplitude',1,'frequency', 10e3, 'duty', 50);

% Save the current settings of the instrument
m.save_settings("instrument_state.mokuconf");
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"filename": "instrument_state.mokuconf"}'\
        http://<ip>/api/waveformgenerator/save_settings
```

</code-block>

</code-group>
