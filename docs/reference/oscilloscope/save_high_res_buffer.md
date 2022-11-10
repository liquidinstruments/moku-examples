---
additional_doc: null
description: Save the high resolution channel buffer data to a file on the Moku's internal storage
method: post
name: save_high_res_buffer
parameters:
- default: ""
  description: Optional comments to be included in output file.
  name: comments
  param_range: null
  type: string
  unit: null
summary: save_high_res_buffer
---
<headers/>


Once completed you can download the file from the device using [download_files](../static/download.md).
<parameters/>

Below are the examples on how to save the high resolution data to a file,

<code-group>
<code-block title="Python">
```python
import json
from moku.instruments import Oscilloscope

i = Oscilloscope('192.168.###.###')
# Configure instrument to desired state
response = i.save_high_res_buffer(comments="Test")
file_name = response["file_name"]
i.download("persist", file_name, "~/high_res_data.li")

```
</code-block>

<code-block title="MATLAB">
```matlab
% Connect to Moku
m = MokuOscilloscope('192.168.###.###', false);
response = m.save_high_res_buffer();
m.download_file('persist', response.file_name, './high_res_data.li')
```
</code-block>

<code-block title="cURL">
```bash
$: FNAME = `curl -H 'Moku-Client-Key: <key>'\
              -H 'Content-Type: application/json'\
              --data '{}'\
              http://<ip>/api/oscilloscope/save_high_res_buffer |
              jq .data.file_name`
$: curl http://<ip>/api/persist/download/$FNAME -o $FNAME
```
</code-block>

</code-group>
