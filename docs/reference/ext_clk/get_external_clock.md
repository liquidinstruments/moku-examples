---
additional_doc: null
description: Check if the Moku is using the external reference clock
method: get
name: get_external_clock
parameters: []
summary: get_external_clock
available_on: "mokupro"
---



<headers/>


<parameters/>

### Examples


<code-group>
<code-block title="Python">
```python
from moku.instruments import Oscilloscope

i = Oscilloscope('192.168.###.###', force_connect=False)
# Here you can access the get_external_clock function
i.get_external_clock()
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuOscilloscope('192.168.###.###', false);

% Here you can access the get_external_clock function
m.get_external_clock()
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        http://<ip>/api/moku/get_external_clock
```
</code-block>

</code-group>

