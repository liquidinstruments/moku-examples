---
additional_doc: Linear interpolation can achieve very fine time resolution between ADC samples by assuming that the waveform is approximately linear in the region of the trigger crossing.
description: Configures interpolation which controls how each event timestamp is determined from the input data around the threshold crossing
method: post
name: set_interpolation
parameters:
    - default: Linear
      description: Interpolation mode
      name: mode
      param_range: None, Linear
      type: string
      unit:
    - default: True
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range:
      type: boolean
      unit:
summary: set_interpolation
---

<headers/>

<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import TimeFrequencyAnalyzer
i = TimeFrequencyAnalyzer('192.168.###.###')
# configure linear interpolation
i.set_interpolation('Linear')
# retrieve data
data = i.get_data()
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuTimeFrequencyAnalyzer('192.168.###.###')
% configure Linear interpolation
m.set_interpolation('Linear')
% retrieve data
m.get_data()
```

</code-block>

<code-block title="cURL">

```bash
# You should create a JSON file with the data content rather than passing
# arguments on the CLI as the lookup data is necessarily very large
$: cat request.json
{
    "mode" : "Linear"
}
$: curl -H 'Moku-Client-Key: <key>' \
    -H 'Content-Type: application/json' \
    --data @request.json \
    http:// <ip >/api/tfa/set_interpolation
```

</code-block>

</code-group>

### Sample response
