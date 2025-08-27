---
additional_doc: null
description: Configures the output waveform for each channel
method: post
name: generate_waveform
parameters:
    - default: null
      description: Target output channel to generate waveform on
      name: channel
      param_range:
          mokugo: 1, 2
          mokulab: 1, 2
          mokupro: 1, 2, 3, 4
          mokudelta: 1, 2, 3, 4, 5, 6, 7, 8
      type: integer
      unit: null
    - default: null
      description:
          "Defines the output sample rate of the AWG. If you don\u2019t specify\
          \ a mode, the fastest output rate for the given data length will be automatically\
          \ chosen. This is correct in almost all circumstances."
      name: sample_rate
      param_range:
          mokugo: Auto, 125Ms, 62.5Ms, 31.25Ms, 15.625Ms
          mokulab: Auto, 1Gs, 500Ms, 250Ms, 125Ms
          mokupro: Auto, 1.25Gs, 625Ms, 312.5Ms
          mokudelta: Auto, 5Gsa, 2.5GSa, 1.25Gs, 625Ms, 312.5Ms
      type: string
      unit: MS/s
    - default: undefined
      description: Lookup table voltage values. The entries of the LUT are normalized to range [-1.0, 1.0]; if the LUT entries are identically zero then it remains unchanged.
      name: lut_data
      param_range: array of -inf, inf
      type: array
      unit: null
    - default: null
      description: Frequency of the waveform
      name: frequency
      param_range:
          mokugo: 1e-3 to 10e6
          mokulab: 1e-3 to 10e6
          mokupro: 1e-3 to 250e6
          mokudelta: 1e-3 to 2e9 (On Moku:Delta platform, sample rates over 625MSa/s are limited to ±500mV)
      type: number
      unit: Hz
    - default: null
      description: Waveform peak-to-peak amplitude (For Moku:Pro, the output voltage is limited to -1V to 1V with a sample rate of 1.25GSa/s. For Moku:Delta, the output voltage is limited to ±500mV with sample rates over 625MSa/s)
      name: amplitude
      param_range: 4e-3 to 10
      type: number
      unit: V
    - default: 0
      description: Waveform phase offset
      name: phase
      param_range: 0 to 360
      type: number
      unit: Deg
    - default: 0
      description: DC offset applied to the waveform
      name: offset
      param_range: -5 to 5
      type: number
      unit: V
    - default: false
      description: Enable linear interpolation of LUT entries.
      name: interpolation
      param_range: null
      type: boolean
      unit: null
    - default: 1MOhm
      description: Waveform load
      name: load
      param_range:
          mokugo: 1MOhm
          mokulab: 50Ohm, 1MOhm
          mokupro: 50Ohm, 1MOhm
          mokudelta: 50Ohm, 1MOhm
      type: string
      unit: null
      deprecated: true
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: generate_waveform
---

<headers/>

The maximum number of points depends on the sample rate:

For Moku:Delta

-   4,096 at 5 GSa/s
-   8,192 at 2.5 GSa/s
-   16,384 at 1.25 GS/s
-   32,768 at 625 MS/s
-   65,536 at 312.5 MS/s

For Moku:Pro

-   16,384 at 1.25 GS/s
-   32,768 at 625 MS/s
-   65,536 at 312.5 MS/s

For Moku:Lab

-   8,192 at 1Gs
-   16,384 at 500Ms
-   32,768 at 250Ms
-   65,536 at 125Ms

For Moku:Go

-   8,192 at 125 MS/s
-   16,384 at 62.5 MS/s
-   32,768 at 31.25 MS/s
-   65,536 at 15.625 MS/s

Values will be normalized to the range [-1.0, +1.0] and then scaled to the desired amplitude and offset.

<parameters/>

### Examples

<code-group>

<code-block title="Python">

```python
import numpy as np
from moku.instruments import ArbitraryWaveformGenerator
t = np.linspace(0, 1, 100)
sq_wave = np.array([-1.0 if x < 0.5 else 1.0 for x in t])
# x = np.linspace(-np.pi, np.pi, 100)
# sine_wave = np.sin(x)
i = ArbitraryWaveformGenerator('192.168.###.###')
i.generate_waveform(channel=1, sample_rate='Auto', lut_data=list(sq_wave),
    frequency=10e3, amplitude=1)
```

</code-block>

<code-block title="MATLAB">

```matlab
%% Prepare the waveforms
% Prepare a square waveform to be generated
t = linspace(0,1,100);
square_wave = sign(sin(2*pi*t));
m = MokuArbitraryWaveformGenerator('192.168.###.###');
% Configure the output waveform in each channel
% Channel 1: sampling rate of 125 MSa/s, square wave, 1kHz, 1Vpp.
m.generate_waveform(1, "125", square_wave, 1e6, 1);
```

</code-block>

<code-block title="cURL">

```bash
# You should create a JSON file with the data content rather than passing
# arguments on the CLI as the lookup data is necessarily very large
$: cat example.json
{
  "channel": 1,
  "sample_rate": "Auto",
  "frequency": 10e3,
  "amplitude": 1,
  "lut_data": [
    # a list of floats between -1 and 1
  ]
}
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data @example.json\
        http://<ip>/api/awg/generate_waveform
```

</code-block>

</code-group>

### Sample response

```json
{
    "amplitude": 1.0,
    "frequency": 10000.0,
    "interpolation": 0,
    "offset": 0.0,
    "phase": 0.0,
    "sample_rate": "125Ms"
}
```
