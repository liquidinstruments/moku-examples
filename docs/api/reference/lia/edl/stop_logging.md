---
additional_doc: Handy method to terminate a in-progress data logging session.
description: Stops the current instrument data logging session.
method: get
name: stop_logging
parameters: []
summary: stop_logging
group: Logger
---

<headers/>
<parameters/>

Partial data log can still be downloaded to local machine using [download_files](../../static/download.md)

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import Datalogger
i = Datalogger('192.168.###.###', force_connect=False)
# Generate a waveform on output channels
# Any other settings...
logFile = i.start_logging(duration=10, comments="Sample script")
time.sleep(5) # Abort the logging session after 5 seconds
i.stop_logging
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuDatalogger('192.168.###.###', true);
% Generate a waveform on output channels
% Any other settings...
logFile = m.start_logging('duration', 10, 'comments', 'Sample script');
m.stop_logging()
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{}'\
        http://<ip>/api/datalogger/stop_logging
```

</code-block>

</code-group>
