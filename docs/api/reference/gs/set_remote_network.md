---
additional_doc: null
description: Configure the remote destination IP, port and MAC address for outgoing packets
method: post
name: set_remote_network
parameters:
    - default: null
      description: IP Address
      name: ip_address
      param_range: null
      type: string
      unit: null
    - default: null
      description: Remote UDP port number
      name: port
      param_range: null
      type: integer
      unit: null
    - default: null
      description: MAC address
      name: mac_address
      param_range: null
      type: string
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_remote_network
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
gs.set_remote_network(ip_address="192.168.1.100", port=4321, mac_address="aa:bb:cc:dd:ee:ff", strict=True)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuMultiInstrument('192.168.###.###', 3)
gs = m.set_instrument(1, @MokuGigabitStreamer)
gs.set_remote_network('192.168.1.100', 4321, 'aa:bb:cc:dd:ee:ff', 'strict', true)
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
       -H 'Content-Type: application/json'\
       --data '{"ip_address": "192.168.1.100","port": 4321, "mac_address":"aa:bb:cc:dd:ee:ff" "strict": true}'\
       http://<ip>/<slot>/api/gs/set_remote_network
```

</code-block>

</code-group>

### Sample response

```json
{
    "ip_address": "192.168.1.100",
    "port": "4321",
    "mac_address": "aa:bb:cc:dd:ee:ff",
}
```
