---
additional_doc:
description: Configure the Digital IO direction
method: post
name: set_dio
parameters:
    - default: nil
      description: A list of maps of pin and direction to configure. Must not be specified at the same time as `direction`
      name: direction_map
      param_range: '[{pin: 1, direction: 1}]'
      type: array
      unit:
    - default: nil
      description: List of 16 DIO directions, one for each pin. Where, 0 represents In and 1 represents Out. Must not be specified at the same time as `direction_map`.
      name: direction
      param_range: '[0/1, ...]'
      type: array
      unit: null
    - default: True
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range:
      type: boolean
      unit:
summary: set_dio
available_on: 'Moku:Go'
---

<headers/>

Digital I/O is configured for Input or Output at the pin level. To use DIO with Multi-instrument mode, the user must complete three steps:

1. Set up the required instrument in the target slot,
2. Configure inputs and/or outputs of that slot to be connected to the Digital I/O, and
3. Configure the direction of the Digital I/O pins individually to meet the needs of the instrument.

The instrument itself cannot govern the pin direction. Pins default to inputs. Pin direction is generally specified using the `direction_map` parameter below which accepts a list of one or more maps, each map sets the direction of a single pin. This can get cumbersome, so a convenience parameter `direction` exists which simply takes a list of exactly 16 0s and 1s, corresponding to input or output (respectively) for each of the 16 pins in the DIO block.

<parameters/>

:::tip NOTE
It is required to connect DIO to a slot before configuring it. Read [set_connections](./set_connections.md)
:::

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import MultiInstrument, WaveformGenerator, Oscilloscope

m = MultiInstrument('192.168.###.###', force_connect=True, platform_id=2)
wg = m.set_instrument(1, WaveformGenerator)
osc = m.set_instrument(2, Oscilloscope)
m.set_dio(direction=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
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
connections = [struct('source', 'Slot1InA', 'destination', 'DIO');
            struct('source', 'Slot1OutA', 'destination', 'Slot2InA')];
m.set_connections(connections);
m.set_dio('direction', [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
```

</code-block>

<code-block title="cURL">

```bash
# You should create a JSON file with the data content rather than passing
# arguments on the CLI as the lookup data is necessarily very large
$: cat request.json
{
  'direction_map':[{'pin':1,'direction':1}]
}
$: curl -H 'Moku-Client-Key: <key>' \
    -H 'Content-Type: application/json' \
    --data @request.json \
    http://<ip>/api/mim/set_dio
```

</code-block>

</code-group>
