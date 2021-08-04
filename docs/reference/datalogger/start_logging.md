---
additional_doc: "Returns name of the log file. By default all the log files are saved to persist directory"
description: "Start the data logging session to file. "
method: post
name: start_logging
parameters:
- default: 60
  description: Duration to log for
  name: duration
  param_range: null
  type: integer
  unit: Seconds
- default: ''
  description: Optional file name prefix
  name: file_name_prefix
  param_range: null
  type: string
  unit: null
- default: ''
  description: Optional comments to be included
  name: comments
  param_range: null
  type: string
  unit: null
- default: false
  description: Pass as true to stop any existing session and begin a new one
  name: stop_existing
  param_range: null
  type: boolean
  unit: null
  warning: Passing true will kill any existing data logging session with out any warning. Use with caution.
summary: start_logging
---

<headers/>

::: warning Caution
To ensure a complete data logging session, it is recommended to track the progress using [logging_progress](logging_progress.md).
:::

<parameters/>

Log files can be downloaded to local machine using [download_files](../static/download.md)

Usage in clients, 

<code-group>
<code-block title="Python">
```python{3}
from moku.instruments import Datalogger
i = Datalogger('192.168.###.###', force_connect=False)
# Generate a waveform on output channels
# Any other settings...
logFile = i.start_logging(duration=10, comments="Sample script")
```
</code-block>

<code-block title="MATLAB">
```matlab{2}
m = MokuDatalogger('192.168.###.###', true);
% Generate a waveform on output channels
% Any other settings...
logFile = m.start_logging('duration', 10, 'comments', 'Sample script');
```
</code-block>
</code-group>


