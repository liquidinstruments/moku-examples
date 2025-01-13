---
additional_doc: null
description: Save the streaming session to a file
method: null
name: stream_to_file
parameters: []
summary: stream_to_file
group: Data Streaming
---

<headers/>
<parameters/>

:::warning Not a HTTP endpoint
This method is available only with Python and MATLAB clients. It uses [mokucli](../../../../cli/moku-cli.md) to stream LI binary data to csv or other possible formats.
:::

`stream_to_file` accepts a single parameter `file_name` which can be any valid name with one of `csv, mat, npy` as extensions.

Examples,

<code-group>
<code-block title="Python">

```python
from moku.instruments import Phasemeter
i = Phasemeter('192.168.###.###')
i.start_streaming(duration=10)
i.stream_to_file() # by default data is streamed to a csv file
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuPhasemeter('192.168.###.###');
m.start_streaming('duration', 10);
m.stream_to_file() % by default data is streamed to a csv file

```

</code-block>

<code-block title="cURL">

```bash
NOT SUPPORTED
```

</code-block>

</code-group>
