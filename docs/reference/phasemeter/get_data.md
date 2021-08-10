---
description: Get all the data from the instrument
method: get
name: get_data
parameters: []
summary: get_data
---


<headers/>
<parameters/>

Everytime the function is called, the data will be returned in the following format

```json
"data":{
  "ch1":{
      "frequency":[],  
      "amplitude":[],
      "phase":[],
      "time":[],   // Will be deprecated in the next release 
      }
  "ch2":{
      "frequency":[],  
      "amplitude":[],
      "phase":[],
      "time":[],   // Will be deprecated in the next release 
      }  
  "ch3":{
      "frequency":[],  
      "amplitude":[],
      "phase":[],
      "time":[],   // Will be deprecated in the next release 
      } 
  "ch4":{
      "frequency":[],  
      "amplitude":[],
      "phase":[],
      "time":[],   // Will be deprecated in the next release 
      } 
}
```

Below are the examples on how to read the data frame,

<code-group>
<code-block title="Python">
```python
from moku.instruments import MokuPhasemeter

i = MokuPhasemeter('192.168.###.###', force_connect=False)

data = i.get_data()
print(data['ch1']['phase'], data['ch2']['frequency'])

```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuPhasemeter('192.168.###.###', false);
data = m.get_data();

disp(data.ch1.phase);
disp(data.ch2.frequency);
disp(data.time);


```
</code-block>
</code-group>



