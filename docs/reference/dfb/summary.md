---
additional_doc: null
description: Returns a short summary of current instrument state
method: get
name: summary
parameters: []
summary: summary
---





<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import DigitalFilterBox
i = DigitalFilterBox('192.168.###.###')
# Following configuration produces Chebyshev type 1 IIR filter
i.set_filter(1, "3.906MHz", shape="Lowpass", type="ChebyshevI")

# Set the probes to monitor Filter 1 and Output 2
i.set_monitor(1, "Filter1")
i.set_monitor(2, "Output1")
i.summary()

```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuDigitalFilterBox('192.168.###.###');
% Following configuration produces Chebyshev type 1 IIR filter
m.set_filter(1, '3.906MHz', 'shape', 'Lowpass', 'type', 'ChebyshevI')
% Set the probes to monitor Filter 1 and Output 2
m.set_monitor(1, 'Filter1')
m.set_monitor(2, 'Output1')
disp(m.summary())
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        http://<ip>/api/digitalfilterbox/summary
```
</code-block>


</code-group>

### Sample response

```plaintext
Moku:Go Digital Filter Box
Input 1, DC coupling, 0 dB attenuation
Input 2, DC coupling, 0 dB attenuation
Control matrix: 1-1 = 1, 1-2 = 0, 2-1 = 0, 2-2 = 1
Filter 1 - 8th-order Butterworth Lowpass, corner frequency 10.00 kHz, sampling rate: 3.9063 MHz
Input offset 0.000 V, input gain +0.0 dB
Output gain +0.0 dB, output offset 0.000 V
Filter 2 - 4th-order Elliptic Bandpass, corner frequencies 100.0 Hz, 10.00 kHz, sampling rate: 3.9063 MHz
Input offset 0.000 V, input gain +0.0 dB
Output gain +0.0 dB, output offset 0.000 V
```