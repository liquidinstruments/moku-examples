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
      description: Acquisition speed (e.g. "30Hz")
      name: acquisition_speed
      param_range:
          mokugo: 30Hz, 119Hz, 477Hz, 1.9kHz, 15.2kHz, 122kHz
          mokulab: 30Hz, 119Hz, 477Hz, 1.9kHz, 15.2kHz
          mokupro: 37Hz, 150Hz, 596Hz, 2.4kHz, 19.1kHz, 152kHz
          mokudelta: 37Hz, 150Hz, 596Hz, 2.4kHz, 19.1kHz, 152kHz
      type: string
      unit: null
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
from moku.instruments import Phasemeter
i = Phasemeter('192.168.###.###', force_connect=True)

### Configure instrument to desired state

# start logging session and read the file name from response
response = i.start_streaming(duration=10)
while True:
  i.get_stream_data()

```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuPhasemeter('192.168.###.###', force_connect=true);

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
        http://<ip>/api/phasemeter/start_streaming
```

</code-block>

</code-group>

### Sample response

```json
{
    "stream_id": "<stream_id>"
}
```
