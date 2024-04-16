---
additional_doc: null
description: Start logging configured intervals
method: post
name: start_logging
parameters:
- default: 
  description: List of event IDs to log (starting from 1)
  name: event_ids
  param_range: 
  type: array
  unit: 
- default: 60
  description: Duration to log for
  name: duration
  param_range: 
  type: number
  unit: 
- default: ''
  description: Optional file name prefix
  name: file_name_prefix
  param_range: 
  type: string
  unit: 
- default: ''
  description: Optional comments to be included
  name: comments
  param_range: 
  type: string
  unit: 
- default: 0
  description: Delay the start by 
  name: delay
  param_range: 
  type: integer
  unit: 
- default: EventTimestamp
  description: The quantity to log (currently only event timestamp is supported
  name: quantity
  param_range: EventTimestamp
  type: string
  unit: 
- default: True
  description: Disable all implicit conversions and coercions.
  name: strict
  param_range: 
  type: boolean
  unit: 
summary: start_logging
---


<headers/>

<parameters/>

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
