---
additional_doc: null
description: Configure outgoing packet maximum transmission unit (MTU)
method: post
name: set_outgoing_packets
parameters:
    - default: null
      description: Network Maximum Transmission Unit
      name: mtu
      param_range: 508bytes, 576bytes, 1500bytes
      type: string
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_outgoing_packets
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
gs.set_outgoing_packets(mtu="1500bytes")
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuMultiInstrument('192.168.###.###', 3)
gs = m.set_instrument(1, @MokuGigabitStreamer)
gs.set_outgoing_packets('1500bytes')
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
       -H 'Content-Type: application/json'\
       --data '{"mtu": "1500bytes"}'\
       http://<ip>/<slot>/api/gs/set_outgoing_packets
```

</code-block>

</code-group>

### Sample response

```json
{
    "line_rate": "5.2424e+09",
    "mtu": "1500bytes",
    "samples_per_channel": "722",
    "udp_payload": "1472",
}
```
