---
title: set_burst_mode | Waveform Generator
additional_doc: null
description: Configures burst modulation on a given channel
method: post
name: set_burst_mode
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
          mokulab: Input1, Input2, Output1, Output2, Internal, External
          mokupro: Input1, Input2, Input3, Input4, Output1, Output2, Output3, Output4, Internal, External
      type: string
      unit: null
    - default: null
      description: Burst mode
      name: mode
      param_range: Gated, Start, NCycle
      type: string
      unit: null
    - default: 0
      description: Trigger threshold level
      name: trigger_level
      param_range:
          mokugo: -5 to 5
          mokulab: -5 to 5
          mokupro: -20 to 20
      type: number
      unit: V
    - default: 3
      description:
          The integer number of signal repetitions to generate once triggered
          (NCycle mode only)
      name: burst_cycles
      param_range: 1 to 1e6
      type: number
      unit: null
    - default: 0.1
      description: Burst duration
      name: burst_duration
      param_range: 1 cycle period to 1e3
      type: number
      unit: Seconds
    - default: 1
      description: Burst Period
      name: burst_period
      param_range: null
      type: number
      unit: null
    - default: undefined
      description: Input Range
      name: input_range
      param_range:
          mokugo: 10Vpp, 50Vpp
          mokulab: 10Vpp, 1Vpp
          mokupro: 400mVpp, 4Vpp, 40Vpp
          mokudelta: 100mVpp, 1Vpp, 10Vpp, 40Vpp
      type: string
      unit: Vpp
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_burst_mode
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import WaveformGenerator

i = WaveformGenerator('192.168.###.###', force_connect=False)
# Generate a sine wave on channel 1, 0.5 Vpp, 5 kHz
# Generate a square wave on channel 2, 1 Vpp, 1 kHz, 50% duty cycle
i.generate_waveform(channel=1, type='Sine', amplitude=0.5, frequency=5e3)
i.generate_waveform(channel=2, type='Square', amplitude=1.0,
                      frequency=1e3, duty=50)
i.set_burst_mode(channel=2, source='Input1', mode='Start',
                     trigger_level=0.4)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuWaveformGenerator('192.168.###.###', false);
% Generate a sine wave on Channel 1
% 1Vpp, 10kHz, 0V offset
m.generate_waveform(1, 'Sine','amplitude', 1, 'frequency',1000,'offset',0.2);
% Generate a sine wave on Channel 2
% 1Vpp, 10kHz, 0V offset, 50% duty cycle
m.generate_waveform(2, 'Sine', 'amplitude',1,'frequency', 10e3);
m.set_burst_mode(2, 'Input1', 'Start', 'trigger_level', 0.4);
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel": 1, "source": "Input1", "mode": "Start", "trigger_level": 0.4}'\
        http://<ip>/api/waveformgenerator/set_burst_mode
```

</code-block>

</code-group>

### Sample response

```json
{
    "burst_cycles": 3,
    "burst_duration": 0.1,
    "burst_period": 1.0,
    "mode": "Gated",
    "source": "Input1",
    "trigger_level": 0.4
}
```
