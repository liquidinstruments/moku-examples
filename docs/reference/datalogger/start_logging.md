---
additional_doc: Returns the computed name and full path of the log file. This can be used later when downloading the file.
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
- default: 0
  description: Delay the logging session by 'n' seconds
  name: delay
  param_range: null
  type: integer
  unit: Seconds
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: start_logging
---

<headers/>

Log files can be downloaded to local machine using [download_files](../static/download.md).


::: warning Caution
To ensure a complete data logging session, it is recommended to track the progress using [logging_progress](logging_progress.md).
:::

<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import Datalogger
i = Datalogger('192.168.###.###')
# Generate a waveform on output channel
i.generate_waveform(channel=1, type='Sine', amplitude=1, frequency=10e3)
i.start_logging(duration=10, comments="Sample_script")
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuDatalogger('192.168.###.###');
% Generate a waveform on output channels
% Any other settings...
logFile = m.start_logging('duration', 10, 'comments', 'Sample script');
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"duration": 10, "comments": "Sample_script"}'\
        http://<ip>/api/datalogger/start_logging
```
</code-block>

</code-group>

### Sample response
```json
{
  "comments":"",
  "duration":10,
  "file_name":"MokuDataLoggerData_20220301_135057.li",
  "file_name_prefix":"MokuDataLoggerData"
}
```
