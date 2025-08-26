---
additional_doc: null
description: Save the high resolution channel buffer data to a file on the Moku's internal storage
method: post
name: save_high_res_buffer
parameters:
    - default: ''
      description: Optional comments to be included in output file.
      name: comments
      param_range: null
      type: string
      unit: null
    - default: 60
      description: Timeout for trigger event if wait_reacquire is true
      name: timeout
      param_range: 0 - inf
      type: number
      unit: Seconds
summary: save_high_res_buffer
group: Monitors
---

<headers/>

Once completed you can download the file from the device using [download_files](../../static/download.md).
The high resolution channel buffer will be saved to `tmp` (Moku:Lab) `persist` (Moku:Go) or `ssd` (Moku:Pro, Moku:Delta).
<parameters/>

Below are the examples on how to save the high resolution data to a file,

<code-group>
<code-block title="Python">

```python
from moku.instruments import DigitalFilterBox
i = DigitalFilterBox('192.168.###.###')
# Configure instrument to desired state
i.set_acquisition_mode(mode="DeepMemory")
response = i.save_high_res_buffer(comments="Test")

# Download the file; "tmp" (Moku:Lab) "persist" (Moku:Go) or "ssd" (Moku:Pro, Moku:Delta).
i.download(response["location"], response["file_name"], "~/high_res_data.li")
```

</code-block>

<code-block title="MATLAB">

```matlab
% Connect to Moku
m = MokuDigitalFilterBox('192.168.###.###');
% Configure instrument to desired state
m.set_acquisition_mode('mode', 'DeepMemory')
result = m.save_high_res_buffer();
% Download the file; 'tmp' (Moku:Lab) 'persist' (Moku:Go) or 'ssd' (Moku:Pro, Moku:Delta).
m.download_file(result.location, response.file_name, './high_res_data.li');
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{}'\
        http://<ip>/api/digitalfilterbox/save_high_res_buffer
```

</code-block>

</code-group>

### Sample response

```json
{
    "file_name": "_tmp_buffer_9cae115e-b1ad-4859-8a29-a6a8af4ab4fd_.li",
    "location": "persist"
}
```
