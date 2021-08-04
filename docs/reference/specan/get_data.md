---
title: get_data
description: Get the latest sweep results
method: get
name: get_data
parameters: []
summary: get_data
---

<headers/>
<parameters/>


Returns a frame of dual-channel frequency spectrum data (amplitude vs frequency in Hz). Every data frame is of **1024** points with following structure

```json
"data":{
  "ch1":[],  // 1024 points
  "ch2":[],  // 1024 points
  "frequency":[],  // 1024 points
}
```

Below are the examples on how to read the data frame,

<code-group>
<code-block title="Python">
```python
from moku.instruments import SpectrumAnalyzer

i = SpectrumAnalyzer('192.168.###.###', force_connect=False)

data = i.get_data()
print(data['ch1'], data['ch2'], data['frequency'])

```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuSpectrumAnalyzer('192.168.###.###', false);
data = m.get_data();

disp(data.ch1);
disp(data.ch2);
disp(data.frequency);


```
</code-block>
</code-group>


