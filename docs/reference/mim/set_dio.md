---
additional_doc: null
description: Configure the Digital IO direction
method: post
name: set_dio
parameters:
- default: 0
  description: List of 16 DIO directions, one for each pin. Where, 0 represents In and 1 represents Out
  name: direction
  param_range: 0, 1
  type: array
  unit: null
- default: undefined
  description: A map or list of map of pin and direction to configure. 
  name: direction_map
  param_range: 
  type: array
  unit: 
- default: True
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: 
  type: boolean
  unit: 
summary: set_dio
available_on: "mokugo"
---


<headers/>
<parameters/>

:::tip NOTE
It is required to connect DIO to a slot before configuring it. Read [set_connections](set_connections.md)
:::

### Examples

<code-group>
<code-block title="Python">
```python

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
