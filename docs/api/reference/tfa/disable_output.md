---
additional_doc: null
description: Disables the given output channel
method: post
name: disable_output
parameters:
    - default:
      description: Target channel
      name: channel
      param_range:
      type: integer
      unit:
    - default: True
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range:
      type: boolean
      unit:
summary: disable_output
---

<headers/>

<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import TimeFrequencyAnalyzer

i = TimeFrequencyAnalyzer('192.168.###.###')
# Disable Out 1
i.disable_output(channel=1)

```

</code-block>

<code-block title="MATLAB">

```matlab
tfa = MokuTimeFrequencyAnalyzer('192.168.###.###');
% Disable Out 1
tfa.disable_output(1);
```

</code-block>

<code-block title="cURL">

```bash
$: cat request.json
{
    "channel" : 1,
    "signal_type":"Interval",
    "scaling":0,
    "zero_point":0
}
$: curl -H 'Moku-Client-Key: <key>' \
    -H 'Content-Type: application/json' \
    --data @request.json \
    http://<ip >/api/tfa/generate_output

```

</code-block>

</code-group>
