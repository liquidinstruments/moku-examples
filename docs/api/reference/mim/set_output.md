---
additional_doc: null
description: Configure the Output gain settings
method: post
name: set_output
parameters:
    - default: True
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range:
      type: boolean
      unit:
    - default:
      description: Target channel
      name: channel
      param_range: 1, 2, 3, 4
      type: integer
      unit:
    - default:
      description: Output Gain
      name: output_gain
      param_range: 0dB, 14dB
      type: string
      unit: dB
summary: set_output
available_on: 'Moku:Pro'
---

<headers/>

When in Multi-instrument Mode, the instruments themselves cannot configure the output gain settings of the Moku. For example, the Waveform Generator instrument is no longer in control of the output gain, as the user may dynamically change the mapping between Waveform Generator output channel and the physical DAC (or not connect it to a DAC at all).

When in Multi-instrument Mode, the user must use this `set_output` function rather than the ones "typically" found in the namespaces of individual instruments.

<parameters/>

:::tip Make Connections First
You must connect an Output (i.e. DAC) to an instrument before configuring its settings. See [set_connections](./set_connections.md) for details of making the connection.
:::

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import MultiInstrument, WaveformGenerator, Oscilloscope

m = MultiInstrument('192.168.###.###', force_connect=True, platform_id=4)
wg = m.set_instrument(1, WaveformGenerator)
osc = m.set_instrument(2, Oscilloscope)
connections = [dict(source="Input1", destination="Slot1InA"),
               dict(source="Slot1OutA", destination="Slot2InA"),
               dict(source="Slot1OutA", destination="Slot2InB"),
               dict(source="Slot2OutA", destination="Output1")]
m.set_connections(connections)
m.set_output(1, "0dB")
```

</code-block>

<code-block title="MATLAB">

```matlab
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
 m.set_output(1, '0dB');
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "output_gain": "0dB"}'\
        http://<ip>/api/mim/set_output
```

</code-block>

</code-group>

### Sample response
