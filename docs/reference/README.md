---
next: /reference/moku/name
---
# Moku API Reference

You can use this API for the command, control and monitoring of
Liquid Instruments **Moku** devices. This interface can supplement or
replace the Windows/Mac interface, allowing the Moku to be scripted
tightly into your next experiment.

:::tip Note
This library only supports interactions with Moku:Go. To command or control Moku:Lab please visit 
- For Python, [PyMoku](https://pypi.org/project/pymoku/)
- For MATLAB, [MATLAB](https://www.liquidinstruments.com/resources/software-utilities/matlab-api/)
:::

## Common Parameters

### Force Connect
Force Connect can be set whenever an instrument object is created, or as a parameter to [claim_ownership](moku/claim_ownership). When `true`, the connection will succeed even if the Moku is currently being accessed by someone else, `false` will return an error in this case. This is the programmatic equivalent of the Moku: App "This Moku is currently in use..." dialog.

In order to correctly track ownership, it is important that API users always finish their sessions with [relinquish_ownership](moku/relinquish_ownership), including from error paths where applicable.

<code-group>
<code-block title="Python">
```python
from moku.instruments import FrequencyResponseAnalyzer

i = FrequencyResponseAnalyzer('192.168.###.###', force_connect=False)
```
</code-block>

<code-block title="MATLAB">
```matlab
m = MokuFrequencyResponseAnalyzer('192.168.###.###', true);
```
</code-block>

<code-block title="cURL">
```bash
$: curl --include
        -H 'Content-Type: application/json'\
        --data '{"force_connect": false}'\
        http://<ip>/api/moku/claim_ownership
```
</code-block>

</code-group>

### Strict Mode

Most of the functions have an additional parameter `strict` which controls coercions of input values. When `strict` is `true` (the default) the Moku API *will not try to coerce* input values to something physically achievable. The API returns an error with appropriate message(s) when it cannot set up the device *exactly* as the user has asked.

For example, if a user asks for a 1GHz sinewave (which is faster than supported by the hardware), then `strict=true` will return an error while `strict=false` will set the Waveform Generator to the *fastest possible* value, and return a human-readable message informing the user what action has been taken.

Disabling strict mode can be useful when setting parameter where "close enough" is acceptable. For example, the precisely supported Edge Times on a Pulse Waveform can be hard to compute beforehand, it's often easier to ask for the "ideal" edge time with `strict=false` then confirm visually that the achieved Edge Time is acceptably close.
