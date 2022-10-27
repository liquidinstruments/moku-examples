---
additional_doc: null
description: Connects the instrument to other instruments, or to external I/O.
method: post
name: set_connections
parameters:
- default: 
  description: List of map of source and destination points
  name: connections
  param_range: 
    mokugo: Source [Input1, Input2, DIO, SlotXOutY]; Destination [Output1, Output2, DIO, SlotXInY]
    mokupro: Source [Input1, Input2, Input3, Input4, SlotXInY]; Destination [Output1, Output2, Output3, Output4, SloutXOutY]
  type: array
  unit: null
summary: set_connections
---

<headers/>

:::tip Incremental Use
The connection action is *incremental*. That is, each connection specified in this function is made in addition to (or in replacement of) an existing connection. There is no way to remove a connection except by replacing it with something else.
:::

:::tip Set Instruments First
Number of inputs and outputs for a given slot are instrument dependent, you must set up the instruments in the target slots before calling this method. See [Getting Started with MiM](../../starting-mim.md) for details.
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
[{source: "InputA", destination: "Slot1InA"}]
$: curl -H 'Moku-Client-Key: <key>' \
    -H 'Content-Type: application/json' \
    --data @request.json \
    http://<ip>/api/mim/set_connections
```
</code-block>

</code-group>

### Sample response
