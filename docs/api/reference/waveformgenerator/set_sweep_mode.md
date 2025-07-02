---
title: set_sweep_mode | Waveform Generator
additional_doc: null
description: Configures sweep modulation on a given channel
method: post
name: set_sweep_mode
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
    - default: null
      description: Trigger source
      name: source
      param_range:
          mokugo: Input1, Input2, Output1, Output2, Internal
          mokulab: Input1, Input2, Output1, Output2, Internal
          mokupro: Input1, Input2, Input3, Input4, Output1, Output2, Output3, Output4, Internal
          mokudelta: Input1, Input2, Input3, Input4, Input5, Input6, Input7, Input8, Output1, Output2, Output3, Output4, Output5, Output6, Output7, Output8, Internal, Ext. trig.
      type: string
      unit: null
    - default: 30000000
      description: Sweep stop Frequency
      name: stop_frequency
      param_range:
          mokugo: 1e-3 to 20e6
          mokulab: 1e-3 to 100e6
          mokupro: 1e-3 to 150e6
          mokudelta: 1e-3 to 2e9
      type: number
      unit: Hz
    - default: 1
      description: Duration of sweep
      name: sweep_time
      param_range: 1 cycle period to 1e3
      type: number
      unit: Seconds
    - default: 0
      description: Trigger threshold level
      name: trigger_level
      param_range:
          mokugo: -5 to 5
          mokulab: -5 to 5
          mokupro: -20 to 20
          mokudelta: -20 to 20
      type: number
      unit: V
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_sweep_mode
---

<headers/>
<parameters/>

<code-group>
<code-block title="Python">

```python
from moku.instruments import WaveformGenerator

i = WaveformGenerator('192.168.###.###')
i.generate_waveform(channel=1, type='Sine', amplitude=0.5, frequency=5e3)
i.generate_waveform(channel=2, type='Sine', amplitude=1.0, frequency=1e6)

# Configure Channel 2 with sweep trigger modulation

# Use Input 1 as trigger source, trigger level is 0.1 V

# Start the sweep at waveform frequency 1 MHz and stop at 10 Hz, each sweep is 3 seconds

i.set_sweep_mode(channel=2, source='Input1', stop_frequency=10.0,
sweep_time=3.0, trigger_level=0.1)

```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuWaveformGenerator('192.168.###.###');
% Generate a sine wave on Channel 1
% 1Vpp, 10kHz, 0V offset
m.generate_waveform(1, 'Sine','amplitude', 1, 'frequency',1000,'offset',0.2);
% Generate a sine wave on Channel 2
% 1Vpp, 10kHz, 0V offset, 50% duty cycle
m.generate_waveform(2, 'Sine', 'amplitude',1,'frequency', 10e3);
m.set_sweep_mode(1, 'Input2', 'stop_frequency', 10e6, 'sweep_time', 0.5, 'trigger_level', 0.1);
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "source": "Input1", "stop_frequency": 10, "sweep_time": 3.0, "trigger_level": 0.1}'\
        http://<ip>/api/waveformgenerator/set_sweep_mode
```

</code-block>

</code-group>

### Sample response

```json
{
    "source": "Input1",
    "stop_frequency": 10.0,
    "sweep_time": 3.0,
    "trigger_level": 0.1
}
```
