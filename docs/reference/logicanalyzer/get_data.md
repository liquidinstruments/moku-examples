---
additional_doc: null
description: Get current sweep data.
method: post
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
available_on: "mokugo"
---


<headers/>
<parameters/>



Every data frame is a time series data of **1024** points with following structure

```json
"data":{
  "pin1":[],  // 1024 points
  "pin2":[],  // 1024 points
  "pin3":[],  // 1024 points
  .
  .
  .
  "pin16":[],  // 1024 points
  "time":[],  // 1024 points
}
```

Below are the examples on how to read the data frame,

<code-group>
<code-block title="Python">
```python
from moku.instruments import LogicAnalyzer
i = LogicAnalyzer('192.168.###.###')
i.set_pin(1, "O")
i.set_pin(2, "H")
i.set_pin(3, "L")
# Configure the output pattern for Pin 1
i.generate_pattern(pin=1, pattern=[1, 0, 0, 0, 0, 0, 0, 0])
i.start_all()
data = i.get_data()
print(data['pin1'], data['pin2'], data['time'])
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuLogicAnalyzer('192.168.###.###');
m.set_pin('pin',1, 'state',"O");
m.set_pin('pin',2, 'state', "H");
m.set_pin('pin',3, 'state', "L");
% Configure the output pattern on Pin 8 to [1 1 0 0]
m.generate_pattern('pin',1,'pattern', [1 1 0 0]);
m.start_all()
data = m.get_data()
disp(data.pin1);
disp(data.pin2);
disp(data.time);
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"wait_reacquire": false, "timeout": 10}'\
        http://<ip>/api/logicanalyzer/get_data |
        jq ".data.pin1"
```
</code-block>

</code-group>
