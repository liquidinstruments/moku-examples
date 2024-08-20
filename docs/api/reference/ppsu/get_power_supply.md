---
additional_doc:
    When using either of the clients, user can access this function directly from
    instrument reference.
description: Get the current state of a specific power supply
method: post
name: get_power_supply
parameters:
    - default: null
      description: ID of the power supply
      name: id
      param_range: null
      type: integer
      unit: null
summary: get_power_supply
available_on: 'Moku:Go'
---

<headers/>

<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import ArbitraryWaveformGenerator

i = ArbitraryWaveformGenerator('192.168.###.###', force_connect=False)
# Here you can access the get_power_supply function
i.get_power_supply(1)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuOscilloscope('192.168.###.###', false);

% Here you can access the get_power_supply function
m.get_power_supply(1)

```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"id": 1}'\
        http://<ip>/api/moku/get_power_supply
```

</code-block>

</code-group>

Sample response for read_power_supply with ID 1,

```json
{
    "id": 1,
    "enabled": true,
    "current_range": [0, 0.15],
    "voltage_range": [-5, 5],
    "set_voltage": 1.99951171875,
    "set_current": 0.1039387308533917,
    "actual_voltage": 1.92825,
    "actual_current": 0.00157257080078127,
    "constant_current_mode": false,
    "constant_voltage_mode": true
}
```
