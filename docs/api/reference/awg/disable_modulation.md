---
additional_doc: null
description: Disable the modulation of a channel
method: post
name: disable_modulation
parameters:
    - default: null
      description: Target channel
      name: channel
      param_range:
          mokugo: 1, 2
          mokulab: 1, 2
          mokupro: 1, 2, 3, 4
      type: integer
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: disable_modulation
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import ArbitraryWaveformGenerator
i = ArbitraryWaveformGenerator('192.168.###.###')
# Configure the output waveform in each channel
# Configure modulation in respective channels

# Disable modulation on channel 2
i.disable_modulation(2)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuArbitraryWaveformGenerator('192.168.###.###');
% Configure the output waveform in each channel
% Configure modulation in respective channels

% Disable modulation on channel 2
m.disable_modulation(2)

```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel":2}'\
        http://<ip>/api/awg/disable_modulation
```

</code-block>

</code-group>

### Sample Response

```json
{
    "Modulation type": "Off"
}
```
