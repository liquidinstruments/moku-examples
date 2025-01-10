---
additional_doc: null
description: Generate pattern on a single channel
method: post
name: set_pattern_generator
parameters:
    - default: null
      description: Id of pattern generator to generate pattern on
      name: channel
      param_range: 1 to 2
      type: integer
      unit: null
    - default: null
      description: List of pin-pattern mapping, where pin should be a valid pin number and pattern should be a list of Logic high(1) and Logic low(0)
      name: patterns
      param_range: null
      type: array
      unit: null
    - default: null
      description: List of pin/bit to override map, where pin should be a valid pin number and override should be a list of Logic high(1) and Logic low(0)
      name: overrides
      param_range: null
      type: array
      unit: null
    - default: undefined
      description: Rate at which the information is transferred.
      name: baud_rate
      param_range: 125 to 125e6
      type: integer
      unit: null
    - default: undefined
      description:
          Divider to scale down the base frequency of 125 MHz to the tick frequency.
          For example, a divider of 2 provides a 62.5 MHz tick frequency.
      name: divider
      param_range: 1 to 1e6
      type: integer
      unit: null
    - default: 8
      description: Number of ticks
      name: tick_count
      param_range: 1 to 32764
      type: number
      unit: null
    - default: true
      description: Repeat forever
      name: repeat
      param_range: null
      type: boolean
      unit: null
    - default: 1
      description: Number of iterations, valid when repeat is set to false
      name: iterations
      param_range: 1 to 8192
      type: integer
      unit: null
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_pattern_generator
---

<headers/>
<parameters/>

:::tip NOTE

-   **baud_rate** and **divider** are mutually exclusive
-   The pattern is not generated on the Pin unless the state of the pin is set to the corresponding generator, PG1 (or) PG2. Refer [set_pin](./set_pin.md) and [set_pins](./set_pins.md).
:::

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import LogicAnalyzer
i = LogicAnalyzer('192.168.###.###')
patterns = [{"pin": 1, "pattern": [0, 1, 0, 1]},
            {"pin": 16, "pattern": [1, 0, 1, 0]}]
# Configure PG1 to generate pattern on pins 1 and 16
i.set_pattern_generator(1, patterns=patterns, divider=12)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLogicAnalyzer('192.168.###.###');
patterns=[struct('pin',1,'pattern',[0,1,0,1]),
          struct('pin',16,'pattern',[1,0,1,0])];
% Configure PG1 to generate pattern on pins 1 and 16
m.set_pattern_generator(1, patterns, 'divider', 12);
```

</code-block>

<code-block title="cURL">

```bash
# If the pattern is longer, consider putting the data in a JSON file
# rather than passing on the command line
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"channel":1,"patterns":[{"pin":1,"pattern":[0,1,0,1]},{"pin":16,"pattern":[1,0,1,0]}],"divider":12}'\
        http://<ip>/api/logicanalyzer/set_pattern_generator
```

</code-block>

</code-group>

### Sample response

```json
{
    "divider": 12,
    "iterations": 1,
    "repeat": true,
    "tick_count": 8
}
```
