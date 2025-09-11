---
additional_doc: null
description: Resets the phase accumulator of both output waveforms
method: get
name: sync_phase
parameters: []
summary: sync_phase
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
import numpy as np
from moku.instruments import ArbitraryWaveformGenerator

# Generate the square wave array
t = np.linspace(0, 1, 100)
sq_wave = np.array([-1.0 if x < 0.5 else 1.0 for x in t])

# Connect to your Moku
i = ArbitraryWaveformGenerator('192.168.###.###', force_connect=True)
# Configure the output waveform in both channels
# Sampling rate of 125 MSa/s, square wave, 1MHz, 1Vpp.
i.generate_waveform(channel=1, sample_rate='Auto', lut_data=list(sq_wave),
    frequency=1e6, amplitude=1)
i.generate_waveform(channel=2, sample_rate='Auto', lut_data=list(sq_wave),
    frequency=1e6, amplitude=1)
# Synchronize the phase between the two channels
i.sync_phase()
```

</code-block>

<code-block title="MATLAB">

```matlab
%% Prepare the waveforms
% Prepare a square waveform to be generated
t = linspace(0,1,100);
square_wave = sign(sin(2*pi*t));

%% Moku configuration
% Connect to your Moku
m = MokuArbitraryWaveformGenerator('192.168.###.###', force_connect=true);
% Configure the output waveform in both channels
% Sampling rate of 125 MSa/s, square wave, 1MHz, 1Vpp.
m.generate_waveform(1, "125", square_wave, 1e6, 1);
m.generate_waveform(2, "125", square_wave, 1e6, 1);

% Synchronize the phase between the two channels
m.sync_phase();
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        http://<ip>/api/awg/sync_phase
```

</code-block>

</code-group>
