---
additional_doc: null
description: Synchronize the phase of output waveforms
method: post
name: sync_output_phase
parameters: []
summary: sync_output_phase
---





<headers/>
<parameters/>

<code-group>
<code-block title="Python">
```python
from moku.instruments import Datalogger

i = Datalogger('192.168.###.###')
i.sync_output_phase()
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuDatalogger('192.168.###.###');
m.sync_output_phase();
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{}'\
        http://<ip>/api/datalogger/sync_output_phase
```
</code-block>

</code-group>
