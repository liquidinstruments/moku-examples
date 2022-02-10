---
additional_doc: null
description: Reset the Lock-In Amplifier to default state
method: post
name: set_defaults
parameters: []
summary: set_defaults
---

<headers/>

<parameters/>

### Examples

<code-group>
<code-block title="cURL">
```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{}'\
        http://<ip>/api/lockinamp/set_defaults
```
</code-block>

</code-group>