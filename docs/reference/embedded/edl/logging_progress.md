---
additional_doc: null
description: Returns current logging state.
method: get
name: logging_progress
parameters: []
summary: logging_progress
group: Logger
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
from moku.instruments import Datalogger
i = Datalogger('192.168.###.###', force_connect=False)
# Generate a waveform on output channels
# Any other settings...
logFile = i.start_logging(duration=10, comments="Sample script")
# Track progress percentage of the data logging session
is_logging = True
while is_logging:
    # Wait for the logging session to progress by sleeping 0.5sec
    time.sleep(0.5)
    # Get current progress percentage and print it out
    progress = i.logging_progress()
    remaining_time = int(progress['time_to_end'])
    is_logging = remaining_time > 1
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuDatalogger('192.168.###.###', true);
% Generate a waveform on output channels
% Any other settings...
logFile = m.start_logging('duration', 10, 'comments', 'Sample script');
is_logging = false;
while is_logging
    pause(1);
    progress = m.logging_progress();
    is_logging = progress.time_to_end > 1;
end
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{}'\
        http://<ip>/api/datalogger/logging_progress
```
</code-block>
</code-group>

Sample response, 

```json
{
   "file_name":"MokuDataLoggerData_20210603_101533.li", // Name of the file the data is logged to
   "file_system":"eMMC", // Target file system
   "time_to_end":"2", // Time remaining 
   "bytes_logged":"80", // Bytes logged
   "time_to_start":"0", // If > 0, it is the estimated time remaining 
                        // to begin the data logging session
   "file_format":"Binary" // Format of the file
}
```

