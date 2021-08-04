---
additional_doc: null
description: Get 1 frame of the data from the instrument
method: get
name: get_data
parameters: []
summary: get_data
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
</code-group>