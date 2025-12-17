---
additional_doc: null
description: Configure the local IP address, port and optional multicast address for receiving/transmitting.
method: post
name: set_local_network
parameters:
    - default: null
      description: IP address
      name: ip_address
      param_range: null
      type: string
      unit: null
    - default: null
      description: UDP port number
      name: port
      param_range: null
      type: integer
      unit: null
    - default: not configured
      description: Multicast IP address
      name: multicast_ip_address
      param_range: null
      type: string
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_local_network
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
gs.set_local_network(ip_address='192.168.1.10', port=5000, multicast_ip_address='239.0.0.1', strict=True)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuMultiInstrument('192.168.###.###', 3)
gs = m.set_instrument(1, @MokuGigabitStreamer)
gs.set_local_network('192.168.1.10', 5000, 'multicast_ip_address', '239.0.0.1', "strict": true)
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
       -H 'Content-Type: application/json'\
       --data '{"ip_address": "192.168.1.10", "port": 5000, "multicast_ip_address": "239.0.0.1", "strict" true}'\
       http://<ip>/<slot>/api/gs/set_local_network
```

</code-block>

</code-group>

### Sample response

```json
{
    "ip_address": "192.168.1.10",
    "port": 5000,
    "multicast_ip_address": "239.0.0.1",
}
```
