---
additional_doc: null
description: Set Window function
method: post
name: set_window
parameters:
- default: null
  description: Window Function
  name: window
  param_range: BlackmanHarris, FlatTop, Hanning, Rectangular
  type: string
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_window
---





<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import SpectrumAnalyzer
i = SpectrumAnalyzer('192.168.###.###')
# BlackmanHarris window
i.set_window(window="BlackmanHarris")
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuSpectrumAnalyzer('192.168.###.###');
% BlackmanHarris window
m.set_window('BlackmanHarris');
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"window": "BlackmanHarris"}'\
        http://<ip>/api/spectrumanalyzer/set_window
```
</code-block>


</code-group>

### Sample response
```json
{
  "window": "BlackmanHarris"
}
```