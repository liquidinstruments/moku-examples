---
additional_doc: null
description: Set the state of a single digital I/O pin
method: post
name: set_pin
parameters:

- default: null
  description: Target pin to configure
  name: pin
  param_range: 1 to 16
  type: integer
  unit: null
- default: null
  description: State of the target pin.
  name: state
  param_range: I, O, H, L, X
  type: string
  unit: null
- default: true
  description: Disable all implicit conversions and coercions. Please refer to [Pin Status Definitions](README.md) for the list of available statuses
  name: strict
  param_range: null
  type: boolean
  unit: null
summary: set_pin
available_on: "mokugo"
---


<headers/>
<parameters/>

Please refer to [Pin Status Definitions](README.md) for the list of available statuses

### Examples


<code-group>
<code-block title="Python">
```python
from moku.instruments import LogicAnalyzer
i = LogicAnalyzer('192.168.###.###')
i.set_pin(1, "O")
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuLogicAnalyzer('192.168.###.###');
m.set_pin('pin',1, 'state','O');
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: 17ee67f8476'\
        -H 'Content-Type: application/json'\
        --data '{"pin": 1, "state": "O"}'\
        http://10.1.111.121/api/logicanalyzer/set_pin
```
</code-block>

</code-group>

:::tip Note
set_pin only configures the state of the Pin, to generate a pattern on a pin use
[generate_pattern](generate_pattern.md)
:::
