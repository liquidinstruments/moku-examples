---
description: Get a single chunk out of data stream
method: post
name: get_chunk
parameters:
- default: None
  description: Stream id returned with the start_streaming method
  name: stream_id
  param_range: null
  type: string
  unit: null
summary: get_chunk
---


<headers/>
<parameters/>

:::tip

This method returns the data in the LI binary format. If you are using Python or MATLAB package use `get_stream_data` to get the conveted stream

:::

:::warning
If you are using cURL command, please note that the URL for this method is `http://192.168.###.###/api/v2/get_chunk`. 
Traditional API calls does not have a `/v2` suffix in the URL.
:::

Below are the examples on how to read the data frame,

<code-group>
<code-block title="Python">
```python
from moku.instruments import Oscilloscope

i = Oscilloscope('192.168.###.###')

data = i.get_data()
print(data['ch1'], data['ch2'], data['time'])

```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuOscilloscope('192.168.###.###');
data = m.get_data();

disp(data.ch1);
disp(data.ch2);
disp(data.time);


```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"wait_reacquire": true, "timeout": 10}'\
        http://<ip>/api/oscilloscope/get_data |
        jq ".data.ch1"
```
</code-block>

</code-group>
