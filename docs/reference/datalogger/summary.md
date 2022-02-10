---
additional_doc: null
description: Returns a short summary of current instrument state
method: get
name: summary
parameters: []
summary: summary
---


<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import Datalogger
i = Datalogger('192.168.###.###', force_connect=False)
print(i.summary())
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuDatalogger('192.168.###.###', true);
disp(m.summary());
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{}'\
        http://<ip>/api/datalogger/summary
```
</code-block>

</code-group>

Sample response,

```plaintext
Moku:Go Data Logger
Input 1 - AC coupling, 10 Vpp range
Input 2 - DC coupling, 10 Vpp range
Acquisition rate: 1.0000000000e+02 Hz, Precision mode
```
