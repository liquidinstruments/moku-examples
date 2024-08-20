---
additional_doc: null
description: Stop the current logging session
method: post
name: stop_logging
parameters: []
summary: stop_logging
---

<headers/>

<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import TimeFrequencyAnalyzer
i = TimeFrequencyAnalyzer('192.168.###.###', force_connect=False)
# Configure event detectors, interval analyzers
logFile = i.start_logging(event_ids=[1], duration=10)
i.stop_logging()
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuTimeFrequencyAnalyzer('192.168.###.###');
% Configure event detectors, interval analyzers
m.start_logging([1], 'duration', 10);
m.stop_logging()
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{}'\
        http://<ip>/api/tfa/stop_logging
```

</code-block>

</code-group>
