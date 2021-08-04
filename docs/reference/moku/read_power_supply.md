---
additional_doc: When using either of the clients, user can access this function directly from
                instrument reference.
description: Read current state of a specific power supply
method: post
name: read_power_supply
parameters:
- default: null
  description: ID of the power supply
  name: id
  param_range: null
  type: integer
  unit: null
summary: read_power_supply
available_on: "mokugo"
---



<headers/>


<parameters/>

Examples,

<code-group>
<code-block title="Python">
```python
from moku.instruments import ArbitraryWaveformGenerator

i = ArbitraryWaveformGenerator('192.168.###.###', force_connect=False)
# Here you can access the read_power_supply function
i.read_power_supply(1)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuOscilloscope('192.168.###.###', false);

% Here you can access the read_power_supply function
m.read_power_supply(1)
```
</code-block>
</code-group>


Sample response for read_power_supply with ID 1,

```json
[
   {
      "id":1,
      "enabled":false,
      "current_range":[0, 0.1],
      "voltage_range":[-5, 5],
      "set_voltage":0,
      "set_current":0.1,
      "actual_voltage":-0.05,
      "actual_current":0,
      "constant_current_mode":false,
      "constant_voltage_mode":true
   }
]
```