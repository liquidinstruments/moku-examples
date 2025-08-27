---
additional_doc: null
description: Trigger all channels that are configured for manual triggering.
method: post
name: manual_trigger
parameters: []
summary: manual_trigger
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import WaveformGenerator
i = WaveformGenerator('192.168.###.###')
i.set_burst_mode(channel=2, source='Manual', mode='Start')
i.manual_trigger()
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuWaveformGenerator('192.168.###.###');
m.set_burst_mode(2, 'Manual', 'Start');
m.manual_trigger();
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{}'\
        http://<ip>/api/waveformgenerator/manual_trigger
```

</code-block>

</code-group>
