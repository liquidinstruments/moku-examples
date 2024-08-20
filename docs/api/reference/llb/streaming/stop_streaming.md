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
from moku.instruments import LaserLockBox
import time
i = LaserLockBox('192.168.###.###')
# Configure instrument
i.start_streaming(duration=10)
time.sleep(5) # Abort the streaming session after 5 seconds
i.stop_streaming()
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLaserLockBox('192.168.###.###');
% Configure instrument
m.start_streaming('duration', 10);
pause(5) % Abort the streaming session after 5 seconds
m.stop_streaming()
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        --data '{}'\
        http://<ip>/api/laserlockbox/stop_streaming
```

</code-block>
</code-group>
