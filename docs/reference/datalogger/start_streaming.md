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
- default: undefined
  description: Target samples per second (For Moku:Pro, the maximum sampling rate is limited to 5MSa/s for 2 channel logging and 1.25MSa/s for 3 and 4 channel logging)
  name: sample_rate
  param_range: 
   mokugo: 10 to 1e6
   mokulab: 10 to 1e6
   mokupro: 10 to 10e6
  type: number
  unit: null
- default: true
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: start_streaming
---

<headers/>

<parameters/>

::: tip
The **samplerate** parameter here does exactly the same thing as [set_samplerate](./set_samplerate.md), just in a more convenient way.
:::

### Examples

<code-group>
<code-block title="Python">
```python
import json
from moku.instruments import Datalogger
i = Datalogger('192.168.###.###')

### Configure instrument to desired state

# start logging session and read the file name from response
response = i.start_streaming(duration=10,sample_rate=100e3)
while True:
  i.get_stream_data()

```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuDatalogger('192.168.###.###');

%%% Configure instrument to desired state

m.start_streaming('duration', 10, 'sample_rate', 100e3)


while true
  m.get_stream_data()
end

```
</code-block>


<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"duration": 10, "sample_rate": 100e3}'\
        http://<ip>/api/datalogger/start_streaming
```
</code-block>

</code-group>

### Sample response
```json
{
   "stream_id":"<stream_id>",
}
```
