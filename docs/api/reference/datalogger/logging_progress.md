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

-   **running** → Boolean of running status
-   **complete** → Boolean of completion status
-   **message** → Message of waiting for delay or trigger signal
-   **time_remaining** → Estimated time remaining
-   **samples_logged** → Number of samples logged to file
-   **file_name** → File name of logged file on the device

::: tip INFO
To convert .li binary formatted log files, use liconverter windows app or [mokucli convert](../../../cli/moku-cli.md#mokucli-convert)
:::

<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
import time

from moku.instruments import Datalogger

i = Datalogger('192.168.###.###')

# Configure instrument to desired state

# Start the logging session...

result = i.start_logging(duration=10)
file_name = result['file_name']

# Track progress of the data logging session

complete = False
while complete is False: # Wait for the logging session to progress by sleeping 0.5sec
time.sleep(0.5) # Get current progress and print it out
progress = i.logging_progress()
complete = progress['complete']
if 'time_remaining' in progress:
print(f"Remaining time {progress['time_remaining']} seconds")

```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuDatalogger('192.168.###.###');
%%% Configure instrument to desired state

% start logging session and download file to local directory
m.start_logging('duration',10);

% Set up to display the logging process
progress = m.logging_progress();

% Track the progress of data logging session
while progress.complete < 1
fprintf('%d seconds remaining \n',progress.time_remaining)
pause(1);
progress = m.logging_progress();
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

### Sample response

```json
{
   "complete":False,
   "file_name":"MokuDataLoggerData_20210603_101533.li",
   "running":True,
   "samples_logged":2238,
   "time_remaining":2
}
```
