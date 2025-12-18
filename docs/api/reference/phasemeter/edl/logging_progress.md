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

Log files can be downloaded to local machine using [download_files](../../static/download.md)

Refer to,

-   **running** → Boolean of running status
-   **complete** → Boolean of completion status
-   **message** → Message of waiting for delay or trigger signal
-   **time_remaining** → Estimated time remaining
-   **samples_logged** → Number of samples logged to file
-   **file_name** → File name of logged file on the device

::: tip INFO
To convert .li binary formatted log files, use liconverter windows app or [mokucli convert](../../../../cli/convert.md)
:::

<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
import time
import os

from moku.instruments import Phasemeter
i = Phasemeter('192.168.###.###', force_connect=True)

# Configure instrument to desired state
# Start a 10s logging session
logFile = i.start_logging(duration=10)
file_name = logFile['file_name']

# Track the remaining time of the data logging session
complete = False
while complete is False:
    # Wait for the logging session to progress by sleeping 1 sec
    time.sleep(1)
    # Get the remaining logging duration and print it out
    progress = i.logging_progress()
    complete = progress['complete']
    if 'time_remaining' in progress:
        print(f"Remaining time {progress['time_remaining']} seconds")

# Download the log file from the Moku to the current working directory.
# Moku:Go should be downloaded from "persist", 
# Moku:Delta and Moku:Pro from "ssd", and Moku:Lab from "media'.
# Use liconverter to convert this .li file to .csv
i.download(target="persist", file_name=logFile['file_name'], 
           local_path=logFile['file_name'])
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuPhasemeter('192.168.###.###', force_connect=true);

%%% Configure instrument to desired state
% Start a 10s logging session
logging_request = m.start_logging('duration',10);
log_file = logging_request.file_name;

% Set up to display the logging process
progress = m.logging_progress();

% Track time remaining in data logging session
while progress.complete < 1
    pause(1);
    fprintf('%d seconds remaining \n',progress.time_remaining)
    progress = m.logging_progress();
end
    
%%% Download the log file from the Moku to "Users" folder locally
% Moku:Go should be downloaded from "persist", 
% Moku:Delta and Moku:Pro from "ssd", and Moku:Lab from "media'.
% Use liconverter to convert this .li file to .csv
m.download_file('ssd',log_file,['C:\Users\Users' log_file]);

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
   "complete":False,
   "file_name":"MokuPhasemeterData_20210603_101533.li",
   "message": "Logging in progress, 5 seconds remaining",
   "running":True,
   "samples_logged":2238,
   "time_remaining":5,
   "time_to_start": 0
}
```