---
additional_doc: null
description: Get the converted data stream
method: null
name: get_stream_data
parameters: []
summary: get_stream_data
group: Data Streaming
---

<headers/>
<parameters/>

:::warning Not a HTTP endpoint
This method is available only with Python and MATLAB clients. It uses [mokucli](../../../../cli/) to stream LI binary data to csv or other possible formats.
:::

:::tip
Structure of data stream depends on the instrument configuration. For example, if `Input1` is disabled, returned will be similar to `{'time':[], 'ch2':[]}`
:::

Below are the examples on how to read the data stream,

<code-group>
<code-block title="Python">

```python
from moku.instruments import Phasemeter

i = Phasemeter('192.168.###.###', force_connect=True)

# Start streaming session for 10 seconds
i.start_streaming(duration=10)

# Retrieve the streamed data frame
data = i.get_stream_data()

# Inspect the returned time and channel values
print(data['time'], data['ch1'], data['ch2'])
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuPhasemeter('192.168.###.###', force_connect=true);

% Start streaming session for 10 seconds
m.start_streaming('duration', 10);

% Retrieve the streamed data frame
data = m.get_stream_data();

% Inspect the returned time and channel values
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
