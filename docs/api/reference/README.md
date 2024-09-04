---
next: /api/reference/moku/name
---

# Moku API Reference

You can use this API for the command, control and monitoring of
Liquid Instruments **Moku** devices. This interface can supplement or
replace the Windows/Mac interface, allowing the Moku to be scripted
tightly into your next experiment.

:::tip Note
This library only supports interactions with Moku:Go and Moku:Pro. To command or control Moku:Lab please visit

-   For Python, [PyMoku](https://pypi.org/project/pymoku/)
-   For MATLAB, [MATLAB](https://www.liquidinstruments.com/resources/software-utilities/matlab-api/)
:::

## Common Parameters

### Force Connect

Force Connect can be set whenever an instrument object is created, or as a parameter to [claim_ownership](moku/claim_ownership.md). When `true`, the connection will succeed even if the Moku is currently being accessed by someone else, `false` will return an error in this case. This is the programmatic equivalent of the Moku: App "This Moku is currently in use..." dialog.

In order to correctly track ownership, it is important that API users always finish their sessions with [relinquish_ownership](moku/relinquish_ownership.md), including from error paths where applicable.

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

Most of the functions have an additional parameter `strict` which controls coercions of input values. When `strict` is `true` (the default) the Moku API _will not try to coerce_ input values to something physically achievable. The API returns an error with appropriate message(s) when it cannot set up the device _exactly_ as the user has asked.

For example, if a user asks for a 1GHz sinewave (which is faster than supported by the hardware), then `strict=true` will return an error while `strict=false` will set the Waveform Generator to the _fastest possible_ value, and return a human-readable message informing the user what action has been taken.

Disabling strict mode can be useful when setting parameter where "close enough" is acceptable. For example, the precisely supported Edge Times on a Pulse Waveform can be hard to compute beforehand, it's often easier to ask for the "ideal" edge time with `strict=false` then confirm visually that the achieved Edge Time is acceptably close.

### Range versus Attenuation

All Moku devices have selectable circuits on their inputs that change the amount a signal is attenuated between the input connection and the internal A/D converter. In some instruments, it is most helpful to control this attenuation directly; for others it's more helpful to think of it in terms of the input voltage range that that attenuator allows the Moku to accept without saturating. This is exposed to users as either a `range` or `attenuation` argument to that instrument's `set_frontend` function.

For example: The Oscilloscope exposes `range`. The voltage displayed on the Oscilloscope display is the voltage on the input, regardless of the range setting. Changing range changes the maximum voltage that can be applied to the instrument, and also the voltage resolution of the instrument.

The PID Controller exposes `attenuation`. The voltage displayed on the PID Controller monitors is the voltage on the input, attenuated by the applied attenuation setting. For example, a 1V input on Moku:Pro's `20dB` attenuation setting is shown on the monitors as 0.1V (`20dB` is the attenuation in _power_, voltage goes like the square-root of that, i.e. 20dB is 10x attenuation). Becaus the voltage range _inside_ the instrument is always fixed, the resolution inside the instrument is also fixed.

As a rule: Instruments that measure the inputs or drive the outputs directly display the actual voltage present on the connectors and expose `range` (e.g Oscilloscope, Spectrum Analyzer). Instruments that connect inputs to outputs use `attenuation` as it makes it much easier to calculate the overall gain (e.g. PID Controller, Lock-In Amplifier, Filter Boxes). Again taking the PID Controller as an example: If the input attenuation is `20dB` and the output hardware gain is `14dB`, then your controller needs a Proportional gain of `6dB` to maintain a `0dB` proportional gain from input to output.
