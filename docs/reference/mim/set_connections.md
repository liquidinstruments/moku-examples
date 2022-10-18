---
additional_doc: null
description: Connects the instrument running in one slot with frontend, relays or instrument running in another slot.
method: post
name: set_connections
parameters:
- default: 
  description: List of map of source and destination points
  name: connections
  param_range: 
    mokugo: Input1, Input2, Output1, Output2, DIO
    mokupro: Input1, Input2, Input3, Input4, Output1, Output2, Output3, Output4
  type: array
  unit: null
summary: set_connections
---

<headers/>

:::tip INFO
Number of inputs and outputs for a given slot are instrument dependent, it is required to configure the instrument in slot before calling this method
:::

<parameters/>


To connect `Input1` of the Moku to the first input of the instrument running in `Slot1` request will look something like 
`{"source":"Input1", "destination":"Slot1InA"}`. Similarly, to connect output of instrument running in `Slot2` to `Output1` request will be
`{"source":"Slot2OutA", "destination":"Output1"}`

### Examples

<code-group>
<code-block title="Python">
```python
from moku.instruments import MultiInstrument
m = MultiInstrument('192.168.###.###', force_connect=True, platform_id=2)
wg = m.set_instrument(1, WaveformGenerator)
osc = m.set_instrument(2, Oscilloscope)
connections = [dict(source="Input1", destination="Slot1InA"),
               dict(source="Slot1OutA", destination="Slot2InA"),
               dict(source="Slot1OutA", destination="Slot2InB"),
               dict(source="Slot2OutA", destination="Output1")]
m.set_connections(connections=connections)
```
</code-block>

<code-block title="MATLAB">
```matlab

```
</code-block>

<code-block title="cURL">
```bash
# You should create a JSON file with the data content rather than passing
# arguments on the CLI as the lookup data is necessarily very large
$: cat request.json
{
 
}
$: curl -H 'Moku-Client-Key: <key>'        -H 'Content-Type: application/json'        --data @request.json        
```
</code-block>

</code-group>

### Sample response
