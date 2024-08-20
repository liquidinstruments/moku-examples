---
additional_doc: null
description: Stop sweeping
method: post
name: stop_sweep
parameters: []
summary: stop_sweep
---

<headers/>

Stop sweeping.

This will stop new data frames from being received,
<parameters/>

<code-group>
<code-block title="Python">

```python
from moku.instruments import FrequencyResponseAnalyzer
import time
i = FrequencyResponseAnalyzer('192.168.###.###')
# Set output sweep configuration
i.set_sweep(start_frequency=10e6, stop_frequency=100,
num_points=512, averaging_time=10e-3,
settling_time=10e-3, averaging_cycles=1,
settling_cycles=1)
i.start_sweep()
time.sleep(5) # Delay for 5 seconds
i.stop_sweep()
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuFrequencyResponseAnalyzer('192.168.###.###');
% Set output sweep configuration
m.set_sweep('start_frequency', 10e6, 'stop_frequency', 100, 'num_points', 512, ...
    'averaging_time', 10e-3, 'averaging_cycles', 1,...
    'settling_time', 10e-3, 'settling_cycles', 1);
m.start_sweep()
pause(5) % Delay for 5 seconds
m.stop_sweep()
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{}'\
        http://<ip>/api/fra/stop_sweep
```

</code-block>

</code-group>
