---
title: get_data
description: Get the latest sweep results
method: post
name: get_data
parameters:
    - default: false
      description: Wait for a new trigger event
      name: wait_reacquire
      param_range: null
      type: boolean
      unit: null
    - default: dBm
      description: Units
      name: units
      param_range: dBm, Vrms, Vpp, dBV
      type: string
      unit: null
    - default: false
      description: PSD Units
      name: psdUnits
      param_range: null
      type: boolean
      unit: null
    - default: false
      description: Wait until entire frame is available
      name: wait_complete
      param_range: null
      type: boolean
      unit: null
    - default: 60
      description: Timeout for trigger event if wait_reacquire is true
      name: timeout
      param_range: 0 - inf
      type: number
      unit: Seconds
    - default: false
      description: When set to True, it returns both the raw data for channels and the computed measurements
      name: measurements
      param_range: null
      type: boolean
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: get_data
---

<headers/>
<parameters/>

Returns a frame of dual-channel frequency spectrum data (amplitude vs frequency in Hz). Every data frame has the following structure. Please note that the number of data points for each frame can be any number between **512** and **1024**.

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

i = SpectrumAnalyzer('192.168.###.###')

data = i.get_data()
print(data['ch1'], data['ch2'], data['frequency'])

```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuSpectrumAnalyzer('192.168.###.###');
data = m.get_data();

disp(data.ch1);
disp(data.ch2);
disp(data.frequency);

```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"wait_reacquire": true, "timeout": 10}'\
        http://<ip>/api/spectrumanalyzer/get_data |
        jq ".data.ch1"
```

</code-block>

</code-group>
