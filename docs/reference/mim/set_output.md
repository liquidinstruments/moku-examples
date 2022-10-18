---
additional_doc: null
description: Configure the output relays
method: post
name: set_output
parameters:
- default: True
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: 
  type: boolean
  unit: 
- default: 
  description: Target channel
  name: channel
  param_range: 
  type: integer
  unit: 
- default: 
  description: Output Gain
  name: output_gain
  param_range: 0dB, 14dB
  type: 
  unit: 
summary: set_output
---


<headers/>

<parameters/>

:::tip NOTE
It is required to connect output(s) to a slot before configuring it. Read [set_connections](set_connections.md)
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
