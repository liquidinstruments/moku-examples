---
additional_doc: null
description: Configures the output phase for given channel.
method: post
name: set_output_phase
parameters:
    - default: null
      description: Target channel
      name: channel
      param_range:
          mokugo: 1, 2
          mokulab: 1, 2
          mokupro: 1, 2, 3, 4
          mokudelta: 1, 2, 3, 4, 5, 6, 7, 8
      type: integer
      unit: null
    - default: 0
      description: Output phase difference for the given channel
      name: phase
      param_range: 0 to 360
      type: number
      unit: Degree
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_output_phase
---

<headers/>

<parameters/>

<code-group>
<code-block title="Python">

```python
from moku.instruments import FrequencyResponseAnalyzer

i = FrequencyResponseAnalyzer('192.168.###.###')
# Measure input signal on channel 1
i.fra_measurement(1, input_only=True, start_frequency=100,
                  stop_frequency=20e6, averaging_cycles=1)
# Set output phase
i.set_output_phase(1, 180)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuFrequencyResponseAnalyzer('192.168.###.###');
% Measure input signal on channel 1
m.fra_measurement(1, 'input_only', true, 'start_frequency', 100,
                  'stop_frequency', 20e6, 'averaging_cycles', 1)
% Set output phase
m.set_output_phase(2, 180)
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "phase": 180}'\
        http://<ip>/api/fra/set_output_phase
```

</code-block>

</code-group>

### Sample response

```json
{ "phase": 180.0 }
```
