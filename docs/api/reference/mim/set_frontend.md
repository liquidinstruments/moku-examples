---
additional_doc: null
description: Configure the Analog front-end settings of the Moku when in Multi-instrument Mode.
method: post
name: set_frontend
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
      param_range:
          mokugo: 1, 2
          mokulab: 1, 2
          mokupro: 1, 2, 3, 4
          mokudelta: 1, 2, 3, 4, 5, 6, 7, 8
      type: integer
      unit:
    - default:
      description: Impedance
      name: impedance
      param_range:
          mokugo: 1MOhm
          mokulab: 50Ohm, 1MOhm
          mokupro: 50Ohm, 1MOhm
          mokudelta: 50Ohm, 1MOhm
      type: string
      unit:
    - default:
      description: Input coupling
      name: coupling
      param_range: AC, DC
      type: string
      unit: null
    - default:
      description: Input attenuation
      name: attenuation
      param_range:
          mokugo: 0dB, -14dB
          mokulab: 0dB, -20dB
          mokupro: 0dB, -20dB, -40dB
          mokudelta: 20dB, 0dB, -20dB, -32dB
      type: string
      unit:
summary: set_frontend
---

<headers/>

When in Multi-instrument Mode, the instruments themselves cannot configure the front-end settings of the Moku. For example, the Oscilloscope instrument is no longer in control of the input attenuation/range, as that input may be shared by multiple instruments.

When in Multi-instrument Mode, the user must use this `set_frontend` function rather than the ones "typically" found in the namespaces of individual instruments.

<parameters/>

:::tip Make Connections First
You must connect an Input (i.e. ADC) to an instrument before configuring its settings. See [set_connections](./set_connections.md) for details of making the connection.
:::

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import MultiInstrument, WaveformGenerator, Oscilloscope

m = MultiInstrument('192.168.###.###', force_connect=True, platform_id=2)
wg = m.set_instrument(1, WaveformGenerator)
osc = m.set_instrument(2, Oscilloscope)
m.set_frontend(1, "1MOhm", "DC", "0dB")
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuMultiInstrument('10.1.111.210', 2, true);
%% Configure the instruments
% WaveformGenerator in slot1
% SpectrumAnalyzer in slot2
wg = m.set_instrument(1, @MokuWaveformGenerator);
sa = m.set_instrument(2, @MokuSpectrumAnalyzer);

% configure routing
connections = [struct('source', 'Input1', 'destination', 'Slot1InA');
struct('source', 'Slot1OutA', 'destination', 'Slot2InA')];
m.set_connections(connections);
% configure frontend
m.set_frontend(1, '1MOhm', 'DC', '0dB');

```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "impedance": "1MOhm", "coupling": "AC", "attenuation": "0dB"}'\
        http://<ip>/api/mim/set_frontend
```

</code-block>

</code-group>

### Sample response
