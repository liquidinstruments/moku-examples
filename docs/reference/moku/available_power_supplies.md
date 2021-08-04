---
description: Get all available power supplies on the Moku hardware
method: get
additional_doc: When using either of the clients, user can access this function directly from
                instrument reference.
name: available_power_supplies
parameters: []
summary: available_power_supplies
available_on: "mokugo"
---

<headers/>

Examples,

<code-group>
<code-block title="Python">
```python
from moku.instruments import ArbitraryWaveformGenerator

i = ArbitraryWaveformGenerator('192.168.###.###', force_connect=False)
# Here you can access the available_power_supplies function
i.available_power_supplies()
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuOscilloscope('192.168.###.###', false);

% Here you can access the available_power_supplies function
m.available_power_supplies()
```
</code-block>
</code-group>

Sample response,

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
   // other power supplies..
]
```
<parameters/>