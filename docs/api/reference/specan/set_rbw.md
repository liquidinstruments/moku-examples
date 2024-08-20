---
additional_doc: Actual resolution bandwidth will be
    rounded to the nearest allowable unit when settings are applied to the device.
description: Set desired Resolution Bandwidth.
method: post
name: set_rbw
parameters:
    - default: null
      description: Resolution bandwidth mode
      name: mode
      param_range: Auto, Manual, Minimum
      type: string
      unit: null
    - default: 5000
      description: Desired RBW value (only in manual mode)
      name: rbw_value
      param_range: null
      type: number
      unit: Hz
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_rbw
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import SpectrumAnalyzer
i = SpectrumAnalyzer('192.168.###.###')
i.set_rbw('Auto')
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuSpectrumAnalyzer('192.168.###.###');
m.set_rbw('Auto');
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"mode": "Auto"}'\
        http://<ip>/api/spectrumanalyzer/set_rbw
```

</code-block>

</code-group>

### Sample response

```json
{
    "mode": "Auto"
}
```
