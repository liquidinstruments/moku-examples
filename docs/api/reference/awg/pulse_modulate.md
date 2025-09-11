---
additional_doc: The Arbitrary Waveform Generator has the ability to insert a dead time between cycles of  the look-up table. This time is specified in cycles of the waveform. During this time, the output will be held at the given dead_voltage. This allows the user to, for example, generate infrequent pulses without using space in the LUT to specify the time between, keeping the full LUT size to provide a high-resolution pulse shape.
description: Configures pulse modulation mode of a channel.
method: post
name: pulse_modulate
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
      description: Number of cycles which show the dead voltage.
      name: dead_cycles
      param_range: 1 to 262144
      type: number
      unit: null
    - default: 0
      description: Signal level during dead time (the voltage cannot be below low level or above high level)
      name: dead_voltage
      param_range: -5 to 5
      type: number
      unit: V
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: pulse_modulate
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
import numpy as np
from moku.instruments import ArbitraryWaveformGenerator

# Generate the square wave array with 100 points, half of the cycle is at -1.0 and the other half at 1.0
t = np.linspace(0, 1, 100)
sq_wave = np.array([-1.0 if x < 0.5 else 1.0 for x in t])

# Connect to your Moku
i = ArbitraryWaveformGenerator('192.168.###.###', force_connect=True)

# Generate the square wave output Channel 1 at 10 kHz with an amplitude of 1 Vpp
i.generate_waveform(channel=1, sample_rate='Auto', lut_data=list(sq_wave),
    frequency=10e3, amplitude=1)

# Add a pulse modulation to channel 1, with 2 dead cycles at 0V between each cycle of the waveform
i.pulse_modulate(1, dead_cycles=2, dead_voltage=0)
```

</code-block>

<code-block title="MATLAB">

```matlab
%% Prepare the waveforms
% Prepare a square waveform to be generated
t = linspace(0,1,100);
square_wave = sign(sin(2*pi*t));

%% Configure your Moku
% Connect to your Moku and deploy the Arbitrary Waveform Generator
m = MokuArbitraryWaveformGenerator('192.168.###.###', force_connect=true);
% Configure the output waveform in each channel
% Channel 1: sampling rate of 125 MSa/s, square wave, 1kHz, 1Vpp.
m.generate_waveform(1, "125", square_wave, 1e6, 1);
% Add a pulse modulation to channel 1, with 2 dead cycles at 0V between each cycle of the waveform
m.pulse_modulate(1,'dead_cycles',2,'dead_voltage',0);
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel":1, "dead_cycles": 2, "dead_voltage": 0}'\
        http://<ip>/api/awg/pulse_modulate
```

</code-block>

</code-group>

### Sample response

```json
{
    "dead_cycles": 100,
    "dead_voltage": 0.5
}
```
