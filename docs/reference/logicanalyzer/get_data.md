---
additional_doc: null
description: Get current sweep data.
method: get
name: get_data
parameters: []
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
```python{9,10}
from moku.instruments import LogicAnalyzer
i = LogicAnalyzer('192.168.###.###', force_connect=False)
i.set_pin("Pin1", "O")
i.set_pin("Pin1", "H")
i.set_pin("Pin1", "L")
# Configure the output pattern for Pin 1
i.generate_pattern("Pin1", [1, 0, 0, 0, 0, 0, 0, 0])
i.start_all()
data = i.get_data()
print(data['pin1'], data['pin2'], data['time'])
```
</code-block>

<code-block title="MATLAB">
```matlab{8-10}
m = MokuLogicAnalyzer('192.168.###.###', true);
m.set_pin("Pin1", "O");
m.set_pin("Pin1", "H");
m.set_pin("Pin1", 'L');
% Configure the output pattern on Pin 8 to [1 1 0 0]
m.generate_pattern('Pin1', [1 1 0 0]);
m.start_all()
disp(data.pin1);
disp(data.pin2);
disp(data.time);
```
</code-block>
</code-group>
