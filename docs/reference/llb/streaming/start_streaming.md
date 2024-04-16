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
  description: Acquisition mode
  name: mode
  param_range: Normal, Precision, DeepMemory, PeakDetect
  type: string
  unit: null
- default: 1000
  description: Acquisition rate
  name: rate
  param_range: null
  type: integer
  unit: null
- default: undefined
  description: Trigger Source
  name: trigger_source
  param_range: 
    mokugo: ProbeA, ProbeB
    mokulab: ProbeA, ProbeB, External
    mokupro: ProbeA, ProbeB, ProbeC, ProbeD, External
  type: string
  unit: null
- default: 0
  description: Trigger level
  name: trigger_level
  param_range: -5 to 5
  type: number
  unit: V
summary: start_streaming
group: Data Streaming
---

<headers/>

<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
import json
from moku.instruments import LaserLockBox
i = LaserLockBox('192.168.###.###')

### Configure instrument to desired state

# start logging session and read the file name from response
response = i.start_streaming(duration=10)
while True:
  i.get_stream_data()

```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuLaserLockBox('192.168.###.###');

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
        http://<ip>/api/laserlockbox/start_streaming
```
</code-block>

</code-group>

### Sample response
```json
{
   "stream_id":"<stream_id>",
}
```
