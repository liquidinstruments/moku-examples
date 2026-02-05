---
next: /api/reference/moku/name
---

# Moku API Reference

You can use this API for the command, control and monitoring of
Liquid Instruments **Moku** devices. This interface can supplement or
replace the Windows/Mac interface, allowing the Moku to be scripted
tightly into your next experiment.

## Common Parameters

### Force Connect

Force Connect can be set whenever an instrument object is created, or as a parameter to [claim_ownership](./moku/claim_ownership.md). When `true`, the connection will succeed even if the Moku is currently being accessed by someone else, `false` will return an error in this case. This is the programmatic equivalent of the Moku: App "This Moku is currently in use..." dialog.

In order to correctly track ownership, it is important that API users always finish their sessions with [relinquish_ownership](./moku/relinquish_ownership.md), including from error paths where applicable.

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

As a rule: Instruments that measure the inputs or drive the outputs directly display the actual voltage present on the connectors and expose `range` (e.g Oscilloscope, Spectrum Analyzer). Instruments that connect inputs to outputs use `attenuation` as it makes it much easier to calculate the overall gain (e.g. PID Controller, Lock-in Amplifier, Filter Boxes). Again taking the PID Controller as an example: If the input attenuation is `20dB` and the output hardware gain is `14dB`, then your controller needs a Proportional gain of `6dB` to maintain a `0dB` proportional gain from input to output.

### Bandwidth

The selectable input bandwidth setting is only available for Moku:Pro, offering either 300 MHz or 600 MHz, which can be set independently for each channel. Moku:Delta's input bandwidth is tied to the input impedance:

-   50 Ohm impedance: 2 GHz bandwidth
-   1 MOhm impedance: 1 MHz bandwidth

There are tradeoffs to each bandwidth range, decide based on the properties which best suit your application (and input impedance, if required). It is best to use the lowest bandwidth that encompasses the bandwidth of the signal.

Generally, lower bandwidth modes reduce access to higher-frequency content but provide a lower noise floor and therefore better signal-to-noise ratio (SNR). Conversely, higher bandwidth modes extend frequency response but increase the broadband noise floor, resulting in reduced SNR. Read more about how [bandwidth selection affects your measurements](https://knowledge.liquidinstruments.com/en_US/mokupro-input-bandwidth).

Moku:Pro’s 5 GSa/s ADC is shared across all inputs, giving each channel a 1.25 GSa/s sampling rate and a corresponding Nyquist limit of 625 MHz. The 600 MHz input lowpass filter helps prevent higher-frequency signals from aliasing into the measurement band.

Some instruments, like the Oscilloscope and Spectrum Analyzer, oversample to improve spectral purity, which previously limited the Spectrum Analyzer and Phasemeter to 300 MHz. With the updated bandwidth selection, users can now extend their range to 600 MHz by adjusting the input bandwidth settings.

For signals below 300 MHz, the 300 MHz bandwidth should be used to minimize noise. For example, when measuring a 100 MHz signal, the lower-bandwidth filter produces a noticeably reduced noise floor above 300 MHz while preserving signal amplitude.

It is not required to set the bandwidth on any instrument, however can be set using an instrument's `set_frontend()` function, with the notation `bandwidth="600MHz"` (Python) and `'bandwidth', '300MHz'` (MATLAB).
