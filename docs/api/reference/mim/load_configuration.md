---
additional_doc: null
description: Load a Multi-Instrument Mode `.mokuconf` configuration from a file. 
method: post
name: load_configuration
parameters:
    - default: null
      description: The path to load the `.mokuconf` file from.
      name: filename
      param_range: null
      type: string
      unit: null
summary: load_configuration
---

<headers/>

To create a saved configuration, use either `save_configuration` or the desktop app.
See [save_configuration](./save_configuration.md) to save the configuration.

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
m = MultiInstrument('192.168.###.###', force_connect=True)

# Load the saved configuration of Multi-Instrument Mode from a .mokuconf file
m.load_configuration(filename="mim_state.mokuconf")

# To load the instrument settings, set instruments in the
# slots, then use load_settings
wg = m.set_instrument(1, WaveformGenerator)
osc = m.set_instrument(2, Oscilloscope)
# Load the instrument settings
wg.load_settings(filename="wg_state.mokuconf")
osc.load_settings(filename="osc_state.mokuconf")
```

</code-block>

<code-block title="MATLAB">

```matlab
% Connect to your Moku
m = MokuMultiInstrument('192.168.###.###', force_connect=true);

% Load the saved configuration of Multi-Instrument Mode from a .mokuconf file
m.load_configuration('mim_state.mokuconf');

% To load the instrument settings, set instruments in the
% slots, then use load_settings
wg = m.set_instrument(1, @MokuWaveformGenerator);
osc = m.set_instrument(2, @MokuOscilloscope);
% Load the instrument settings
wg.load_settings('wg_state.mokuconf');
osc.load_settings('osc_state.mokuconf');
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"filename": "mim_state.mokuconf"}'\
        http://<ip>/api/mim/load_configuration
```

</code-block>

</code-group>
