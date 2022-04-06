---
description: Get all available power supplies on the Moku hardware
method: get
additional_doc: When using either of the clients, user can access this function directly from
                instrument reference.
name: get_power_supplies
parameters: []
summary: get_power_supplies
available_on: "mokugo"
---

<headers/>

### Examples


<code-group>
<code-block title="Python">
```python
from moku.instruments import ArbitraryWaveformGenerator

i = ArbitraryWaveformGenerator('192.168.###.###', force_connect=False)
# Here you can access the get_power_supplies function
i.get_power_supplies()
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuOscilloscope('192.168.###.###', false);

% Here you can access the get_power_supplies function
m.get_power_supplies()
```
</code-block>

<code-block title="cURL">
```bash
$: curl http://<ip>/api/moku/get_power_supplies
```
</code-block>

</code-group>

Sample response,

```json
[
   {
      "id":1,
      "enabled":true,
      "current_range":[0, 0.15],
      "voltage_range":[-5, 5],
      "set_voltage":1.99951171875,
      "set_current":0.1039387308533917,
      "actual_voltage":1.928500000000001,
      "actual_current":0.001562500000000022,
      "constant_current_mode":false,
      "constant_voltage_mode":true
   }
   // other power supplies..
]
```
<parameters/>