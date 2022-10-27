---
additional_doc: null
description: Configure the Output gain settings
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
  param_range: 1, 2, 3, 4
  type: integer
  unit: 
- default: 
  description: Output Gain
  name: output_gain
  param_range: 0dB, 14dB
  type: 
  unit: 
summary: set_output
available_on: "mokupro"
---

<headers/>

When in Multi-instrument Mode, the instruments themselves cannot configure the output gain settings of the Moku. For example, the Waveform Generator instrument is no longer in control of the output gain, as the user may dynamically change the mapping between Waveform Generator output channel and the physical DAC (or not connect it to a DAC at all).

When in Multi-instrument Mode, the user must use this `set_output` function rather than the ones "typically" found in the namespaces of individual instruments.

<parameters/>

:::tip Make Connections First
You must connect an Output (i.e. DAC) to an instrument before configuring its settings. See [set_connections](set_connections.md) for details of making the connection.
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
