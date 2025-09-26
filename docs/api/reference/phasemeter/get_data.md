---
description: Get the most recent amplitude, frequency and phase reading
method: post
name: get_data
parameters:
    - default: false
      description: Wait for a new trigger event
      name: wait_reacquire
      param_range: null
      type: boolean
      unit: null
      deprecated: true
    - default: 60
      description: Timeout for trigger event if wait_reacquire is true
      name: timeout
      param_range: 0 - inf
      type: number
      unit: Seconds
      deprecated: true

summary: get_data
---

<headers/>
<parameters/>

Every time the function is called, the data will be returned in the following format

```json
{
  "data":{
    "ch1":{
      "frequency":...,
      "amplitude":...,
      "phase":...,

    },
    "ch2":{
      "frequency":...,
      "amplitude":...,
      "phase":...,

    },
    "ch3":{
      "frequency":...,
      "amplitude":...,
      "phase":...,

    },
    "ch4":{
      "frequency":...,
      "amplitude":...,
      "phase":...,

    }
  }
}
```

Below are the examples on how to read the data frame,

<code-group>
<code-block title="Python">

```python
from moku.instruments import Phasemeter

i = Phasemeter('192.168.###.###', force_connect=True)

data = i.get_data()
print(data['ch1']['phase'], data['ch2']['frequency'])

```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuPhasemeter('192.168.###.###', force_connect=true);
data = m.get_data();

disp(data.ch1.phase);
disp(data.ch2.frequency);
disp(data.time);

```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"wait_reacquire": true, "timeout": 10}'\
        http://<ip>/api/phasemeter/get_data |
        jq ".data.ch1.phase"
```

</code-block>

</code-group>
