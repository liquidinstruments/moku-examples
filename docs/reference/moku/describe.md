---
additional_doc: When using the Python or MATLAB clients, user can access this function directly from
                instrument reference.
description: Returns hardware version, firmware version and API Server version
method: get
name: describe
parameters: []
summary: describe
---


<headers/>

### Examples


<code-group>
<code-block title="Python">
```python
from moku.instruments import ArbitraryWaveformGenerator

i = ArbitraryWaveformGenerator('192.168.###.###', force_connect=False)

# Here you can access the describe function
i.describe()
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuOscilloscope('192.168.###.###', false);

% Here you can access the describe function
m.describe()
```
</code-block>

<code-block title="cURL">
```bash
$: curl http://<ip>/api/moku/describe
```
</code-block>

</code-group>

Sample response,

```json
{
    'firmware': 'xxx', //Firmware version Moku running on 
    'version': '1.3beta', //API Server version
    'hardware': 'Moku:Go' 
}
```