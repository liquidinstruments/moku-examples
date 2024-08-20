---
additional_doc: Handy method to terminate a in-progress data streaming session.
description: Stops the current instrument data streaming session.
method: post
name: stop_streaming
parameters: []
summary: stop_streaming
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import Datalogger
import time
i = Datalogger('192.168.###.###')
# Generate a waveform on output channel
i.generate_waveform(channel=1, type='Sine', amplitude=1, frequency=10e3)
i.start_streaming(duration=10)
time.sleep(5) # Abort the streaming session after 5 seconds
i.stop_streaming()
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuDatalogger('192.168.###.###');
% Generate a waveform on output channels
m.generate_waveform(1, 'Sine', 'amplitude',1, 'frequency',10e3);
m.start_streaming('duration', 10);
pause(5) % Abort the streaming session after 5 seconds
m.stop_streaming()
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        --data '{}'\
        http://<ip>/api/datalogger/stop_streaming
```

</code-block>
</code-group>
