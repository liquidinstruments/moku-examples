---
additional_doc: When successful, returns the log file name and the configuration the session is requested with.
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
- default: Normal
  description: Acquisition mode
  name: mode
  param_range: Normal, Precision, DeepMemory, PeakDetect
  type: string
  unit: null
- default: 1000
  description: Acquistion rate
  name: rate
  param_range: null
  type: integer
  unit: Hz
summary: start_logging
group: Logger

---

<headers/>

Log files can be downloaded to local machine using [download_files](../static/download.md).


::: warning Caution
It is recommended to track the progress of data logging session before relinquishing the ownership [logging_progress](logging_progress.md).
:::

<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
import json
from moku.instruments import LockInAmp
i = LockInAmp('192.168.###.###')

### Configure instrument to desired state

# set probe points
i.set_monitor(1, "Input1")

# start logging session and read the file name from response
response = json.loads(i.start_logging(duration=10))
i.start_logging(duration=10, comments="Sample_script")
file_name = response["file_name"]

# download file to local directory
i.download("persist", file_name, f"~/Desktop/{file_name}")
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuLockInAmp('192.168.###.###');
%%% Configure instrument to desired state

% set probe points
m.set_monitor(1, "Input1");

% start logging session and download file to local directory
response = jsondecode(m.start_logging('duration',10));
m.download_file('persist', response.file_name, strcat('<path to download>', ...
    response.file_name));
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"duration": 10, "comments": "Example"}'\
        http://<ip>/api/lockinamp/start_logging
```
</code-block>

</code-group>

### Sample response,
```json
{
   "acquisition_mode":"Normal",
   "comments":"",
   "duration":10,
   "file_name":"MokuLockInAmpData_20220408_141414.li",
   "rate":1000.0
}
```



