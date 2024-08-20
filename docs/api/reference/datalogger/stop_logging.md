---
additional_doc: Handy method to terminate a in-progress data logging session.
description: Stops the current instrument data logging session.
method: get
name: stop_logging
parameters: []
summary: stop_logging
---

<headers/>
<parameters/>

Partial data log can still be downloaded to local machine using [download_files](../static/download.md)

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import Datalogger
import time
i = Datalogger('192.168.###.###')
# Generate a waveform on output channel
i.generate_waveform(channel=1, type='Sine', amplitude=1, frequency=10e3)
i.start_logging(duration=10, comments="Sample_script")
time.sleep(5) # Abort the logging session after 5 seconds
i.stop_logging()
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuDatalogger('192.168.###.###');
% Generate a waveform on output channels
m.generate_waveform(1, 'Sine', 'amplitude',1, 'frequency',10e3);
m.start_logging('duration', 10, 'comments', 'Sample_script');
pause(5) % Abort the logging session after 5 seconds
m.stop_logging()
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        http://<ip>/api/datalogger/stop_logging
```

</code-block>
</code-group>
