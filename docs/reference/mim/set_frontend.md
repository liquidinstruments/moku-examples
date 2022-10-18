---
additional_doc: null
description: Configure the multi instrument mode frontend
method: post
name: set_frontend
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
    mokugo: 1, 2
    mokupro: 1, 2, 3, 4
  type: integer
  unit: 
- default: 
  description: Impedance
  name: impedance
  param_range:
   mokugo: 1MOhm
   mokupro: 50Ohm, 1MOhm  
  type: string
  unit: 
- default: 
  description: Input coupling
  name: coupling
  param_range: AC, DC
  type: string
  unit: null
- default: 
  description: Input attenuation
  name: attenuation
  param_range: 
    mokugo: 0dB, -14dB
    mokupro: -20dB, -40dB
  type: string
  unit: 
summary: set_frontend
---


<headers/>

<parameters/>

:::tip NOTE
It is required to connect frontend to a slot before configuring it. Read [set_connections](set_connections.md)
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
