---
additional_doc: Returns the computed name and full path of the log file. This can be used later when downloading the file.
description: 'Start the data logging session to file. '
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
      description: Acquisition speed 
      name: acquisition_speed
      param_range:
          mokugo: 30Hz, 119Hz, 477Hz, 1.9kHz, 15.2kHz, 122kHz
          mokulab: 30Hz, 119Hz, 477Hz, 1.9kHz, 15.2kHz
          mokupro: 37Hz, 150Hz, 596Hz, 2.4kHz, 19.1kHz, 152kHz
          mokudelta: 37Hz, 150Hz, 596Hz, 2.4kHz, 19.1kHz, 152kHz
      type: string
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: start_logging
---

<headers/>

Log files can be downloaded to local machine using [download_files](../../static/download.md).

::: warning Tip
To ensure a complete data logging session, it is recommended to track the progress using [logging_progress](./logging_progress.md).
:::

<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
import time

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

% Track the progress of data logging session
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
        --data '{"duration": 10, "comments": "Sample_script"}'\
        http://<ip>/api/lockinamp/start_logging
```

</code-block>

</code-group>

### Sample response

```json
{
    "Save to": 2, 
    "Start": 0, 
    "comments": "",
    "duration": 10,
    "rate": 1000.0,
    "file_name": "MokuPhasemeterData_20220301_135057.li",
    "file_name_prefix": "MokuDataLoggerData",
    "location": "ssd"
}
```

