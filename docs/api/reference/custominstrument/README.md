---
title: Moku Custom Instrument
prev: false
---

# Moku Custom Instrument

Provides access to Moku Custom Instrument generated with [Moku Compile](https://www.liquidinstruments.com/moku-compile/)

If you are using the Custom Instrument instrument directly through the REST API, the instrument name as used in the URL is `custominstrument`.

:::warning Multi-Instrument Mode
Moku Custom Instruments can only be used in Multi-Instrument Mode. Refer to the [Multi-Instrument Mode Getting Started Guide](../../getting-started/starting-mim.md) for more details.
:::

### Deploying the instrument 

<code-group>
<code-block title="Python">

```python
from moku.instruments import MultiInstrument, CustomInstrument
m = MultiInstrument('192.168.###.###', platform_id=2)
bitstream = "path/to/project/adder/bitstreams.tar.gz"
mc = m.set_instrument(1, CustomInstrument, bitstream=bitstream)
```

</code-block>

<code-block title="MATLAB">

```matlab
m = MokuMultiInstrument('192.168.###.###', 'platform_id', 2);
bitstream = 'path/to/project/adder/bitstreams.tar';
mc = m.set_instrument(1, @MokuCustomInstrument, bitstream);

```

</code-block>

:::tip Loading your Bitstream
You can load the Moku Compile generated bitstream onto your Moku either when setting the instrument (as above) or with the [file upload](../static/upload.md) function.
:::

<!-- <code-block title="cURL">

```bash
$: curl -H 'Moku-Client-Key: <key>'\
        -H 'Content-Type: application/json'\
        --data '{"filename": "instrument_state.mokuconf"}'\
        http://<ip>/api/slot1/custominstrument/load_settings
```

</code-block> -->

</code-group>

<function-index/>
