---
additional_doc: If enabled, the get_data function will return a new key named xcorr containing the cross-correlation data in its response.
description: Enables cross correlation between requested channels
method: post
name: enable_xcorr
parameters:
- default: True
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: 
  type: boolean
  unit: 
- default: 
  description: Source channel A
  name: channel_a
  param_range: 
    mokugo: Input1, Input2, InputA, InputB
    mokulab: Input1, Input2, InputA, InputB
    mokupro: Input1, Input2, Input3, Input4, InputA, InputB
  type: 
  unit: 
- default: 
  description: Source channel B
  name: channel_b
  param_range: 
    mokugo: Input1, Input2, InputA, InputB
    mokulab: Input1, Input2, InputA, InputB
    mokupro: Input1, Input2, Input3, Input4, InputA, InputB
  type: 
  unit: 
summary: enable_xcorr
---


<headers/>

<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import SpectrumAnalyzer
i = SpectrumAnalyzer("192.168.###.###")
i.enable_xcorr("Input1", "Input2")
data = i.get_data()
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuSpectrumAnalyzer("192.168.###.###")
m.enable_xcorr("Input1", "Input2")
data = m.get_data()
```
</code-block>

<code-block title="cURL">
```bash
# You should create a JSON file with the data content rather than passing
# arguments on the CLI as the lookup data is necessarily very large
$: cat request.json
{
    "channel_a": "Input1",
    "channel_b": "Input2"
}
$: curl -H 'Moku-Client-Key: <key>' \
    -H 'Content-Type: application/json' \
    --data @request.json \
    http:// <ip >/api/spectrumanalyzer/enable_xcorr        
```
</code-block>

</code-group>

