---
additional_doc: While event statistics show the count and interval statistics of the acquisition period. The histogram is a collection of time measurements.
description: Get statistics and histogram for each interval analyzer
method: post
name: get_data
parameters:
    - default: 60
      description: How long to wait for data to be acquired
      name: timeout
      param_range:
      type: number
      unit:
    - default: True
      description: Disable all implicit conversions and coercions
      name: strict
      param_range:
      type: boolean
      unit:
summary: get_data
---

<headers/>

<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import TimeFrequencyAnalyzer
i = TimeFrequencyAnalyzer('192.168.###.###')
# Configure event detectors
# Configure interval analyzers
# retrieve data
data = i.get_data()
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuTimeFrequencyAnalyzer('192.168.###.###')
% Configure event detectors
% Configure interval analyzers
% retrieve data
m.get_data()
```

</code-block>

<code-block title="cURL">

```bash
# You should create a JSON file with the data content rather than passing
# arguments on the CLI as the lookup data is necessarily very large
$: cat request.json
{
    "timeout":60
}
$: curl -H 'Moku-Client-Key: <key>' \
    -H 'Content-Type: application/json' \
    --data @request.json \
    http:// <ip >/api/tfa/get_data

```

</code-block>

</code-group>

### Sample response

```json
{
   "interval1":{
      "histogram":{
         "data":[
            ...
         ],
         "dt":...,
         "t0":...
      },
      "statistics":{
         "count":...,
         "current":...,
         "maximum":...,
         "mean":...,
         "minimum":...
      }
   },
   "interval2":{
      "histogram":{
         "data":[
            ...
          ],
         "dt":...
         "t0":...
      },
      "statistics":{
         "count":...,
         "current":...,
         "maximum":...,
         "mean":...,
         "minimum":...
      }
   }
}
```
