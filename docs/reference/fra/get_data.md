---
additional_doc: Every data frame is a object representing a frame of dual-channel (amplitude and phase) vs frequency response data.
description: Get current sweep data
method: get
name: get_data
parameters: []
summary: get_data
---





<headers/>
<parameters/>


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
i = FrequencyResponseAnalyzer('192.168.###.###', force_connect=False)

data = i.get_data()
# Print out the data for Channel 1
print(frame['ch1']['frequency'], frame['ch1']['magnitude'],
          frame['ch1']['phase'])
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuFrequencyResponseAnalyzer('192.168.###.###', false);
data = m.get_data();
disp(data.ch1.frequency);
disp(data.ch1.magnitude);
disp(data.ch1.phase);


```
</code-block>
</code-group>
