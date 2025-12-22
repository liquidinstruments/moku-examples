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

Partial data log can still be downloaded to local machine using
[download_files](../../static/download.md)

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import Phasemeter
i = Phasemeter('192.168.###.###', force_connect=True)
# Start a 10s logging session
logFile = i.start_logging(duration=10, comments="Sample script")
# Abort the logging session after 5 seconds
time.sleep(5) 
i.stop_logging
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuPhasemeter('192.168.###.###', force_connect=true);
% Start a 10s logging session
log_file = m.start_logging('duration', 10, 'comments', 'Sample script');
% Abort the logging session after 5 seconds
pause(5);
m.stop_logging();
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
