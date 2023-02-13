---
additional_doc: Returns the streaming interface id which holds the data stream
description: "Start a streaming session"
method: post
name: start_streaming
parameters:
- default: undefined
  description: Duration to stream for
  name: duration
  param_range: null
  type: integer
  unit: Seconds
summary: start_streaming
---

<headers/>

::: warning Caution
Data stream should be retreived as soon as the streaming session begins, not retreiving will push the streaming session into error state. You can kow the status of streaming session through [get_stream_status](getters.md).
:::

<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
import json
from moku.instruments import Datalogger
i = Datalogger('192.168.###.###')

### Configure instrument to desired state

# start logging session and read the file name from response
response = i.start_streaming(duration=10)
while True:
  i.get_stream_data()

```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuDatalogger('192.168.###.###');

%%% Configure instrument to desired state

m.start_streaming()

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
