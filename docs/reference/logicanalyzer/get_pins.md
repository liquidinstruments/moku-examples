---
additional_doc: null
description: Get current state of all Digital I/O pins
method: post
name: get_pins
parameters: []
summary: get_pins
available_on: "mokugo"
---

<headers/>
<parameters/>

<code-group>
<code-block title="Python">
```python{3}
from moku.instruments import LogicAnalyzer
i = LogicAnalyzer('192.168.###.###', force_connect=False)
i.get_pins()
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuLogicAnalyzer('192.168.###.###', true);
m.get_pins();
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{}'\
        http://<ip>/api/logicanalyzer/get_pins
```
</code-block>

</code-group>

Sample response,

```json
{
   "Pin 1":"I",
   "Pin 2":"I",
   "Pin 3":"I",
   "Pin 4":"I",
   "Pin 5":"O",
   "Pin 6":"I",
   "Pin 7":"H",
   "Pin 8":"H",
   "Pin 9":"I",
   "Pin 10":"O",
   "Pin 11":"I",
   "Pin 12":"I",
   "Pin 13":"I",
   "Pin 14":"O",
   "Pin 15":"I",
   "Pin 16":"H",
}
```

