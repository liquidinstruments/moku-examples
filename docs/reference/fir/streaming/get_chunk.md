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

This method returns the data in the LI binary format. If you are using Python or MATLAB package use [get_stream_data](get_stream_data.md) to get the conveted stream

:::

:::warning
If you are using cURL command, please note that the URL for this method is `http://192.168.###.###/api/v2/get_chunk`. 
Traditional API calls does not have a `/v2` suffix in the URL.
:::

Examples,

<code-group>
<code-block title="Python">
```python
from moku.instruments import FIRFilterBox

i = FIRFilterBox('192.168.###.###')
i.start_streaming(duration=10)
data = i.get_chunk()

```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuFIRFilterBox('192.168.###.###');
m.start_streaming('duration', 10);
data = m.get_chunk();
```
</code-block>

<code-block title="cURL">
```bash
$: curl -data '{"stream_id": "<stream_id>"}'\
        http://<ip>/api/v2/get_chunk > data.li
```
</code-block>

</code-group>
