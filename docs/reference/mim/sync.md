---
additional_doc: Calling this function is the equivalent of pressing the "sync" button in the UI
description: Synchronize the phase of internal oscillators and across all the instrument slots
method: get
name: sync
parameters: []
summary: sync
---

<headers/>

When running multiple instruments simultaneously, it is often required that one or more of them have a common concept of phase. For example, a Waveform Generator generating a modulation signal, and a Lock-in Amplifier demodulating it, should share the concept of "zero phase".

This is accomplished by configuring each instrument individually, then calling this `sync` function to simultaneously reset all phase counters.

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
