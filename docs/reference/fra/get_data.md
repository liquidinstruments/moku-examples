---
additional_doc: Every data frame is a object representing a frame of dual-channel (amplitude and phase) vs frequency response data.
description: Get current sweep data
method: post
name: get_data
parameters:
- default: false
  description: Wait for a new trigger event
  name: wait_reacquire
  param_range: null
  type: boolean
  unit: null
- default: false
  description: Wait until complete frame is available
  name: wait_complete
  param_range: null
  type: boolean
  unit: null
- default: 60
  description: Timeout for trigger event if wait_reacquire is true
  name: timeout
  param_range: 0 - inf
  type: integer
  unit: Seconds
summary: get_data
---





<headers/>
<parameters/>

:::tip NOTE
When the single sweep mode is enabled, it is recommended to set the `timeout` parameter to a value greater than the *estimated_sweep_time*, failing to do this will raise a timeout error. *estimated_sweep_time* is returned as part of response in `set_sweep` and `start_sweep` functions. It can also be retrieved using `get_sweep` function.
:::

Every data frame is of **1024** points with following structure,

```json
"data":{
  "ch1":[
    "magnitude":[],
    "phase":[],
    "frequency":[]
  ],
  "ch2":[
    "magnitude":[],
    "phase":[],
    "frequency":[]
  ],
}
```

Below are the examples on how to read the data frame,

<code-group>
<code-block title="Python">
```python
from moku.instruments import FrequencyResponseAnalyzer
i = FrequencyResponseAnalyzer('192.168.###.###')

data = i.get_data()
# Print out the data for Channel 1
print(frame['ch1']['frequency'], frame['ch1']['magnitude'],
          frame['ch1']['phase'])
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuFrequencyResponseAnalyzer('192.168.###.###');
data = m.get_data();
disp(data.ch1.frequency);
disp(data.ch1.magnitude);
disp(data.ch1.phase);


```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"wait_reacquire": true, "timeout": 10}'\
        http://<ip>/api/fra/get_data |
        jq ".data.ch1.magnitude"
```
</code-block>

</code-group>
