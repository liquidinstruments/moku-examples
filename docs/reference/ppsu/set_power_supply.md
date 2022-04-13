---
additional_doc: null
description: Configure a specific power supply channel
method: post
name: set_power_supply
parameters:
- default: null
  description: Target power supply
  name: id
  param_range: null
  type: integer
  unit: null
- default: true
  description: Enable/Disable power supply
  name: enable
  param_range: null
  type: boolean
  unit: null
- default: 3
  description: Voltage set point
  name: voltage
  param_range: null
  type: number
  unit: null
- default: 0.1
  description: Current set point
  name: current
  param_range: null
  type: number
  unit: null
summary: set_power_supply
available_on: "mokugo"
---

<headers/>

Once a load is connected, the power supply operates either at the set current or set voltage, whichever reaches the set point first. set_power_supply returns the same response as [read_power_supply](./read_power_supply.md)

<parameters/>

When using either of the clients, user can access this function directly from
instrument reference.

### Examples


<code-group>
<code-block title="Python">
```python
from moku.instruments import ArbitraryWaveformGenerator

i = ArbitraryWaveformGenerator('192.168.###.###', force_connect=False)
# Here you can access the set_power_supply function
i.set_power_supply(1, enable=True, voltage=5, current=0.2)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuOscilloscope('192.168.###.###', false);

% Here you can access the set_power_supply function
m.set_power_supply(1, 'enable', 'true', 'voltage', 5, 'current', 0.2)
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"id": 1, "enable": true, "voltage": 5, "current": 0.2}'\
        http://<ip>/api/datalogger/set_acquisition_mode
```
</code-block>

</code-group>

### Sample response,
```json
{
   "id":1,
   "enabled":true,
   "current_range":[0, 0.15],
   "voltage_range":[-5, 5],
   "set_voltage":5,
   "set_current":0.1,
   "actual_voltage":4.93525,
   "actual_current":0.001582641601562518,
   "constant_current_mode":false,
   "constant_voltage_mode":true
}
```