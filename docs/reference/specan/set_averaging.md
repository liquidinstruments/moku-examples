---
additional_doc: null
description: Enables frame averaging at the given frame rate.
method: post
name: set_averaging
parameters:
- default: True
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: 
  type: boolean
  unit: 
- default: 0.1
  description: Target frame duration
  name: target_duration
  param_range: 
  type: number
  unit: 
summary: set_averaging
---


<headers/>

<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import SpectrumAnalyzer
i = SpectrumAnalyzer("192.168.###.###")
# configure spectrum analyzer
i.set_averaging(target_duration=0.1)
data = i.get_data()
```
</code-block>

<code-block title="MATLAB">

```matlab
m = MokuSpectrumAnalyzer("192.168.###.###")
% configure spectrum analyzer
m.set_averaging('target_duration', 0.1)
m.get_data()
```
</code-block>

<code-block title="cURL">
```bash
# You should create a JSON file with the data content rather than passing
# arguments on the CLI as the lookup data is necessarily very large
$: cat request.json
{
    "target_duration": 0.1
}
$: curl -H 'Moku-Client-Key: <key>' \
    -H 'Content-Type: application/json' \
    --data @request.json \
    http:// <ip >/api/tfa/set_averaging    
```
</code-block>

</code-group>

### Sample response
