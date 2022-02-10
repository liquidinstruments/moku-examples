---
additional_doc: This is commonly the first operation when connecting to a Moku. It can be used to check whether someone else owns the device, take ownership of it, and retrieve a client key. This function is called *implicitly* from most wrappers when the instrument object is first created.
description: Declare ownership of the Moku and retrieve a client key
method: post
name: claim_ownership
parameters:
- default: false
  description: Claim ownership even if the device is already owned
  name: force_connect
  param_range: true, false
  type: bool
  unit: null
summary: Claim the ownership of Moku
---

<headers/>
<parameters/>

:::warning REST API Only
This operation is called implicitly from Python and MATLAB libraries when the instrument objects are created. The user only needs to call this themselves if they're using the REST API directly. Refer to the [REST Getting Started Guide](/starting-curl.html) for more information.
:::

<code-group>
<code-block title="cURL">
```bash
$: curl --include \
        --data '{}'\
        -H 'Content-Type: application/json'\
        http://<ip>/api/moku/claim_ownership

HTTP/1.1 200 OK
...
Moku-Client-Key: 17cfd311ecb

{"success":true,"data":null,"code":null,"messages":null}
```
</code-block>
</code-group>
