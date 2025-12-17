---
additional_doc: null
description: Returns a short summary of current instrument state
method: get
name: summary
parameters: []
summary: summary
available_on: 'Moku:Delta'
---

<headers/>
<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import MultiInstrument, GigabitStreamer 
m = MultiInstrument('192.168.###.###', platform_id=3)
gs = m.set_instrument(1, GigabitStreamer)
print(gs.summary())
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuMultiInstrument('192.168.###.###', 3)
gs = m.set_instrument(1, @MokuGigabitStreamer)
gs.summary()
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
       -H 'Content-Type: application/json'\
       http://<ip>/<slot>/api/gs/summary
```

</code-block>

</code-group>

### Sample response

```text
Moku:Delta Gigabit Streamer
Enabled inputs: Input 1, Input 2, Input 3, Input 4
Sample rate 39.0625 MSa/s, decimation factor 128, Normal mode, 16 bit samples
Local network: IP 192.168.74.1 (multicast not configured), port 4991, MAC 12:34:56:78:9A:BC
Outgoing packets: MTU 1500 bytes, 176 Sa per channel, UDP payload 1,436 bytes, line rate 2.624 Gbit/s
Remote network: IP 192.168.75.1, port 4991, MAC 1A:2B:3C:4D:5E:6F
Receiving: off
Interpolation: Linear
Output 1 - gain +0.0 dB, offset 0.000 0 V, output disabled
Output 2 - gain +0.0 dB, offset 0.000 0 V, output disabled
Output 3 - gain +0.0 dB, offset 0.000 0 V, output disabled
Output 4 - gain +0.0 dB, offset 0.000 0 V, output disabled
Clock blending - External 10 MHz frequency reference disabled, 1 pps GNSS synchronization reference disabled

```
