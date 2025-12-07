---
additional_doc: null
description: Save Multi-Instrument Mode configuration to a file. The file name should have a `.mokuconf` extension to be compatible with other tools.
method: post
name: save_configuration
parameters:
    - default: null
      description: The path to save the `.mokuconf` file to.
      name: filename
      param_range: null
      type: string
      unit: null
summary: save_configuration
---

<headers/>

Calling `save_configuration` will write the current configuration of the Multi-Instrument Mode to the provided `filename`.

Extension should be `.mokuconf`. This configuration can be loaded using [`load_configuration`](./load_configuration.md).

This will currently only save the configuration that can be configured through this class. Each instrument must also be saved individually.

Multi-Instrument Mode configuration includes:

-   Slot count and sample rate.
-   Instrument placement.
-   Connections.
-   Analog input, DIO, and analog output settings.

It does not include the settings of the instruments in each slot.

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

# Save the current configuration of Multi-Instrument Mode
m.save_configuration(filename="mim_state.mokuconf")

# Use save_settings to save the settings of the instruments in the slots
wg.save_settings(filename="wg_state.mokuconf")
osc.save_settings(filename="osc_state.mokuconf")
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

% Save the current configuration of Multi-Instrument Mode
m.save_configuration('mim_state.mokuconf');

% Use save_settings to save the settings of the instruments in the slots
wg.save_settings('wg_state.mokuconf');
osc.save_settings('osc_state.mokuconf');
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"filename": "mim_state.mokuconf"}'\
        http://<ip>/api/mim/save_configuration
```

</code-block>

</code-group>
