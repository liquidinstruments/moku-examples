---
additional_doc: null
description: Get the converted data stream
method: null
name: get_stream_data
parameters: []
summary: get_stream_data
---

<headers/>
<parameters/>

:::warning Not a HTTP endpoint
This method is available only with Python and MATLAB clients. It uses [mokucli](../../../cli/moku-cli) to stream LI binary data to csv or other possible formats.
:::

:::tip
Structure of data stream depends on the instrument configuration. For example, if `Input1` is disabled, returned will be similar to `{'time':[], 'ch2':[]}`
:::

Below are the examples on how to read the data stream,

<code-group>
<code-block title="Python">

```python
from moku.instruments import Datalogger
i = Datalogger('192.168.###.###')
i.start_streaming(duration=10)
data = i.get_stream_data()
# Print out the data
print(data['time'], data['ch1'], data['ch2'])
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuDatalogger('192.168.###.###');
m.start_streaming('duration', 10);
data = m.get_stream_data();
disp(data.time);
disp(data.ch1);
disp(data.ch2);

```

</code-block>

<code-block title="cURL">

```bash
NOT SUPPORTED
```

</code-block>

</code-group>
