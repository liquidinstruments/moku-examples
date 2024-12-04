---
description: Get single frame of the data from the instrument
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
    - default: false
      description: When set to True, it returns both the raw data for channels and the computed measurements
      name: measurements
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
---

<headers/>
<parameters/>

Every data frame is a time series data of points with following structure

```json
"data": {
  "ch1":  [],
  "ch2":  [],
  "time": [],
}
```

Each element of the data has the same length, set by the `max_length` parameter of `set_timebase` and
defaulting to 1024 points. Note that depending on the timebase set and the hardware version, the achievable
number of points varies. It will never be fewer than half the number of points requested, and is usually
within a few percent of requested. This means that user code must be able to process frames whose lengths
vary with different timebases.

The returned timebase is guaranteed to include the requested timebase, but may have a small number of
extra samples before or after again due to achievable decimation rates. If the precise bounds are important,
the user code should trim the returned data to align with the requested timebase.

Below are the examples on how to read the data frame,

<code-group>
<code-block title="Python">

```python
from moku.instruments import Oscilloscope

i = Oscilloscope('192.168.###.###')

data = i.get_data()
print(data['ch1'], data['ch2'], data['time'])

```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuOscilloscope('192.168.###.###');
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
        --data '{"wait_reacquire": true, "timeout": 10}'\
        http://<ip>/api/oscilloscope/get_data |
        jq ".data.ch1"
```

</code-block>

</code-group>
