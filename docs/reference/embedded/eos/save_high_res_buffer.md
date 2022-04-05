---
additional_doc: null
description: Save high resolution data to a file
method: get
name: save_high_res_buffer
parameters:
- default: ""
  description: Optional comments to be included in output file.
  name: comments
  param_range: null
  type: string
  unit: null
summary: save_high_res_buffer
group: Monitors
---
<headers/>

Once completed you can download the file from the device using [download_files](../static/download.md).
<parameters/>

Below are the examples on how to read the data frame,

<code-group>
<code-block title="Python">
```python
from moku.instruments import PIDController
i = PIDController('192.168.###.###', force_connect=False)
data = i.get_data()
print(data['ch1'], data['ch2'], data['time'])
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuPIDController('192.168.###.###', false);
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
        --data '{}'\
        http://<ip>/api/pid/get_data |
        jq ".data.ch1"
```
</code-block>

</code-group>