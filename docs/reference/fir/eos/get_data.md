---
additional_doc: In the PID instrument, the frame comes from the Monitor subsystem. See `set_monitor`.
description: Get a frame of the data from the instrument
method: get
name: get_data
parameters: 
- default: null
  description: Wait for a new trigger event
  name: wait_reacquire
  param_range: null
  type: boolean
  unit: null
- default: 60
  description: Timeout for trigger event if wait_reacquire is true
  name: timeout
  param_range: 0 - inf
  type: number
  unit: Seconds
summary: get_data
group: Monitors
---

<headers/>
<parameters/>

Every data frame is a time series data of **1024** points with following structure

```json
"data":{
  "ch1":[],  // 1024 points
  "ch2":[],  // 1024 points
  "time":[],  // 1024 points
}
```

Below are the examples on how to read the data frame,

<code-group>
<code-block title="Python">
```python
from moku.instruments import FIRFilterBox
i = FIRFilterBox('192.168.###.###')
data = i.get_data()
print(data['ch1'], data['ch2'], data['time'])
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuFIRFilterBox('192.168.###.###');
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
        http://<ip>/api/firfilter/get_data |
        jq ".data.ch1"
```
</code-block>

</code-group>