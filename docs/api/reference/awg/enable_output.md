---
additional_doc: null
description: Enable/disable each output channel
method: post
name: enable_output
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
      description: Enable/disable the specified output channel
      name: enable
      param_range: null
      type: boolean
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: enable_output
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
i.enable_output(1, enable=True)
i.enable_output(2, enable=False)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuArbitraryWaveformGenerator('192.168.###.###');
% Configure the output waveform in each channel
m.enable_output(1, 'enable', true);
m.enable_output(2, 'enable', false);
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel":1, "enable": true}'\
        http://<ip>/api/awg/enable_output
```

</code-block>
</code-group>
