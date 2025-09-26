---
additional_doc: Handy method to terminate a in-progress data streaming session.
description: Stops the current instrument data streaming session.
method: post
name: stop_streaming
parameters: []
summary: stop_streaming
group: Data Streaming
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import Phasemeter
import time
i = Phasemeter('192.168.###.###', force_connect=True)

# Start streaming session for 10 seconds
i.start_streaming(duration=10)
# Abort the streaming session after 5 seconds
time.sleep(5) 
i.stop_streaming()
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuPhasemeter('192.168.###.###', force_connect=true);

% # Start streaming session for 10 seconds
m.start_streaming('duration', 10);
% Abort the streaming session after 5 seconds
pause(5);
m.stop_streaming();
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        --data '{}'\
        http://<ip>/api/phasemeter/stop_streaming
```

</code-block>
</code-group>
