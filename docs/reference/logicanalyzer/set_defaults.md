---
additional_doc: null
description: Set the Logic Analyzer to its default state
method: post
name: set_defaults
parameters: []
summary: set_defaults
available_on: "mokugo"
---

<headers/>

::: tip INFO
Reference to any instrument object will always be in default state.
:::

<parameters/>

<code-group>
<code-block title="Python">
```python
from moku.instruments import LogicAnalyzer
i = LogicAnalyzer('192.168.###.###', force_connect=False)
# LogicAnalyzer reference i is in default state
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuLogicAnalyzer('192.168.###.###', true);
% LogicAnalyzer reference m is in default state
```
</code-block>

<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{}'\
        http://<ip>/api/logicanalyzer/set_defaults
```
</code-block>

</code-group>