---
additional_doc: null
description: Returns current logging state.
method: get
name: logging_progress
parameters: []
summary: logging_progress
---

<headers/>

This method returns a dictionary to track the progress of data logging session. Always call this method in a loop until a desired result is reached.

Log files can be downloaded to local machine using [download_files](../static/download.md)

Refer to,
- **time_to_end** → Estimated time remaining
- **time_to_start** → Time remaining to start the requested session
- **bytes_logged** → Bytes logged to file


::: tip INFO
To convert .li binary formatted log files, use liconverter windows app
:::


<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import Phasemeter
i = Phasemeter('192.168.###.###')
# Configure instrument to desired state

# Start the logging session...
result = json.loads(i.start_logging(duration=10))
file_name = result['file_name']

# Track the progress of data logging session
is_logging = True
while is_logging:
    # Wait for the logging session to progress by sleeping 0.5sec
    time.sleep(0.5)
    # Get current progress percentage and print it out
    progress = json.loads(i.logging_progress())
    remaining_time = int(progress['time_to_end'])
    is_logging = remaining_time > 1

```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuPhasemeter('192.168.###.###');
%%% Configure instrument to desired state

% start logging session and download file to local directory
m.start_logging('duration',10);

% Track the progress of data logging session
is_logging = true;
while is_logging
    progress = jsondecode(m.logging_progress());
    is_logging = progress.time_to_end > 1;
    pause(1);
end
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{}'\
        http://<ip>/api/phasemeter/logging_progress
```
</code-block>
</code-group>

### Sample response

```json
{
   "file_name":"MokuPhasemeterData_20210603_101533.li",
   "time_remaining":9,
   "time_to_end":9,
   "time_to_start":-1,
   "words_logged":24
}
```
