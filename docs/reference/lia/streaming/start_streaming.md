---
additional_doc: Returns the streaming interface id which holds the data stream
description: Start a streaming session
method: post
name: start_streaming
parameters:
- default: undefined
  description: Duration to stream for
  name: duration
  param_range: null
  type: integer
  unit: Seconds
- default: Normal
  description: Acqusition mode
  name: mode
  param_range: Normal, Precision, DeepMemory, PeakDetect
  type: string
  unit: null
- default: 1000
  description: Acqusition rate
  name: rate
  param_range: null
  type: integer
  unit: null
summary: start_streaming
---

<headers/>

<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
import json
from moku.instruments import LockInAmp
i = LockInAmp('192.168.###.###')

### Configure instrument to desired state

# start logging session and read the file name from response
response = i.start_streaming(duration=10)
while True:
  i.get_stream_data()

```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuLockInAmp('192.168.###.###');

%%% Configure instrument to desired state

m.start_streaming('duration', 10)


while true
  m.get_stream_data()
end

```
</code-block>


<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"duration": 10}'\
        http://<ip>/api/lockinamp/start_streaming
```
</code-block>

</code-group>

### Sample response
```json
{
   "stream_id":"<stream_id>",
}
```
