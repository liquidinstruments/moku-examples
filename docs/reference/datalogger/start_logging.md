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
- default: undefined
  description: Trigger Source
  name: trigger_source
  param_range: 
    mokugo: Input1, Input2
    mokulab: Input1, Input2, External
    mokupro: Input1, Input2, Input3, Input4, External
  type: string
  unit: null
- default: 0
  description: Trigger level
  name: trigger_level
  param_range: -5 to 5
  type: number
  unit: V
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
import json
from moku.instruments import Datalogger
i = Datalogger('192.168.###.###')

### Configure instrument to desired state

# start logging session and read the file name from response
response = i.start_logging(duration=10)
i.start_logging(duration=10, comments="Sample_script")
file_name = response["file_name"]

# download file to local directory
i.download("persist", file_name, f"~/Desktop/{file_name}")
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuDatalogger('192.168.###.###');

%%% Configure instrument to desired state

% start logging session and download file to local directory
response = m.start_logging('duration',10);
m.download_file('persist', response.file_name, strcat('<path to download>', ...
    response.file_name));
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
   "acquisition_mode":"Normal",
   "comments":"",
   "duration":10,
   "rate":1000.0
   "file_name":"MokuDataLoggerData_20220301_135057.li",
   "file_name_prefix":"MokuDataLoggerData"
}
```
