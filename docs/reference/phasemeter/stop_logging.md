---
additional_doc: Handy method to terminate a in-progress data logging session.
description: Stops the current instrument data logging session.
method: get
name: stop_logging
parameters: []
summary: stop_logging
available_on: "mokupro"
---


<headers/>
<parameters/>

Partial data log can still be downloaded to local machine using [download_files](../static/download.md)

Usage in clients, 

<code-group>
<code-block title="Python">
```python{6}
from moku.instruments import MokuPhasemeter
i = MokuPhasemeter('192.168.###.###', force_connect=False)
# Generate a waveform on output channels
# Any other settings...
logFile = i.start_logging(duration=10, comments="Sample script")
i.stop_logging()
```
</code-block>

<code-block title="MATLAB">
```matlab{5}
i = MokuPhasemeter('192.168.###.###', true);
% Generate a waveform on output channels
% Any other settings...
logFile = i.start_logging('duration', 10, 'comments', 'Sample script');
i.stop_logging()
```
</code-block>
</code-group>
