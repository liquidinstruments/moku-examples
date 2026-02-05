---
additional_doc: null
description: Configures the demodulation source and optionally its frequency and phase
method: post
name: set_demodulation
parameters:
    - default: null
      description: The demodulation source
      name: mode
      param_range: Internal, External, ExternalPLL, None
      type: string
      unit: null
    - default: 1000000
      description: Frequency of internally-generated demod source
      name: frequency
      type: number
      unit: Hz
    - default: 0
      description: Phase of internally-generated demod source
      name: phase
      type: number
      unit: degrees
    - default: true
      description: Disable all implicit conversions and coercions.
      name: strict
      param_range: null
      type: boolean
      unit: null
summary: set_demodulation
---

<headers/>

## Demodulation source descriptions

### Internal

The `Internal` option allows the use of the local oscillator that is generated internally. This
oscillator utilizes the same timebase as the clock reference.
Note: Moku devices with an external clock reference port can be synchronized with an externally
connected device. Read more about configuring the [external reference clock](../ext_clk/README.md).

### External

The `External` option allows the second input channel to be used as the reference oscillator. This
also allows non-sinusoidal references to be used as the demodulation source, and can be used
to measure correlation or recover specific components of complex input signals. As the external
signal can be an arbitrary shape, it cannot be used to perform dual-phase demodulation, it can
only interrogate one quadrature, `X`.

### External PLL

The `External (PLL)` option allows the second input channel to be used as the reference
oscillator for dual-phase demodulation. The option uses a digitally implemented phase-locked
loop (PLL) to track the phase of the external reference with a user-selectable bandwidth. The
PLL allows the instrument to generate synchronized in-phase and quadrature sinusoids at the
same frequency, with adjustable phase and frequency multipliers. This mode enables the Lock-in
Amplifier to recover both quadrature signals without sharing the same timebase as the external
signal.

### None

The `None` option can be used to bypass the mixing operation, passing the signal directly to
the lowpass filter. This is useful if the necessary signal extraction is done on an external system
or another instrument in Multi-Instrument Mode, and enables modulation-free locking techniques
such as DC locking, side-of-fringe locking, and tilt locking.

<parameters/>

### Examples

<code-group>
<code-block title="Python">

```python
from moku.instruments import LockInAmp
i = LockInAmp('192.168.###.###')
i.set_demodulation(mode="Internal",frequency=1000000,phase=0)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuLockInAmp('192.168.###.###');
m.set_demodulation('mode','Internal','frequency',1000000,'phase',0)
```

</code-block>

<code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"mode":"Internal","frequency":1000000,"phase":0}'\
        http://<ip>/api/lockinamp/set_demodulation
```

</code-block>

</code-group>

### Sample response

```json
{
    "frequency": 1000000.0,
    "mode": "Internal",
    "phase": 0.0
}
```
