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
from moku.instruments import ArbitraryWaveformGenerator
i = ArbitraryWaveformGenerator('192.168.###.###')
i.burst_modulate(2, "Manual", "NCycle", burst_cycles=3)
i.manual_trigger()
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuArbitraryWaveformGenerator('192.168.###.###');
m.burst_modulate(2, "Manual", "NCycle",'burst_cycles',3);
m.manual_trigger();
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{}'\
        http://<ip>/api/awg/manual_trigger
```

</code-block>

</code-group>
