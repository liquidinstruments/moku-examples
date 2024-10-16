---
additional_doc: null
description:
    Configures the specified monitor channel to view the desired Laser Lock Box probe point
    signal.
method: post
name: set_monitor
parameters:
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
    - default: null
      description: Monitor channel
      name: monitor_channel
      param_range:
        mokugo: 1, 2
        mokulab: 1, 2
        mokupro: 1, 2, 3, 4
      type: integer
      unit: null
    - default: null
      description: Monitor channel source.
      name: source
      param_range:
          mokugo: None, LowpassFilter, FastPIDOutput, SlowPIDOutput, ErrorSignal, LocalOscillator, Input1, Input2, Output1, Output2
          mokulab: None, LowpassFilter, FastPIDOutput, SlowPIDOutput, ErrorSignal, LocalOscillator, Input1, Input2, Output1, Output2
          mokupro: None, LowpassFilter, FastPIDOutput, SlowPIDOutput, ErrorSignal, LocalOscillator, Input1, Input2, Output1, Output2, Output3, Output4
      type: string
      unit: null
summary: set_monitor
group: Oscilloscope
---

<headers/>

<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import LaserLockBox
i = LaserLockBox('192.168.###.###')
i.set_aux_oscillator()
# Set the probes to monitor Output 1 and Output 2
i.set_monitor(1, 'Output1')
i.set_monitor(2, 'Output2')

```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLaserLockBox('192.168.###.###');
m.set_aux_oscillator()
% Set the probes to monitor Output 1 and Output 2
m.set_monitor(1, 'Output1')
m.set_monitor(2, 'Output2')
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "source": "Output1"}'\
        http://<ip>/api/laserlockbox/set_monitor
```

</code-block>

</code-group>

### Sample response

```json
{
    "source": "Output1"
}
```
