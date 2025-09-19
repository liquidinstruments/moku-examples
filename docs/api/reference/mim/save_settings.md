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
from moku.instruments import MultiInstrument, WaveformGenerator, Oscilloscope
# Connect to your Moku
m = MultiInstrument('192.168.###.###', force_connect=True, platform_id=2)

# Set instruments in the slots
wg = m.set_instrument(1, WaveformGenerator)
osc = m.set_instrument(2, Oscilloscope)
# Set up the connections
connections = [dict(source="Input1", destination="Slot1InA"),
               dict(source="Slot1OutA", destination="Slot2InA"),
               dict(source="Slot1OutA", destination="Slot2InB"),
               dict(source="Slot2OutA", destination="Output1")]
m.set_connections(connections=connections)

# Save the current settings of Multi-instrument Mode
m.save_settings(filename="instrument_state.mokuconf")
```

</code-block>

<code-block title="MATLAB">

```matlab
% Connect to your Moku
m = MokuMultiInstrument('192.168.###.###', 2);

%% Configure the instruments
% WaveformGenerator in slot1
wg = m.set_instrument(1, @MokuWaveformGenerator);
osc = m.set_instrument(2, @MokuOscilloscope);
% configure routing
connections = [struct('source', 'Input1', 'destination', 'Slot1InA');
            struct('source', 'Slot1OutA', 'destination', 'Slot2InA');
            struct('source', 'Slot1OutA', 'destination', 'Slot2InB');
            struct('source', 'Slot2OutA', 'destination', 'Output1')];
m.set_connections(connections);

% Save the current settings of Multi-instrument Mode
m.save_settings("instrument_state.mokuconf");
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"filename": "instrument_state.mokuconf"}'\
        http://<ip>/api/mim/save_settings
```

</code-block>

</code-group>
