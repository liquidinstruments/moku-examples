---
title: Arbitrary Waveform Generator
prev: false
name: Multi-instrument
description: Arbitrary Waveform Generator in multi-instrument context
---

# Arbitrary Waveform Generator - Multi-instrument mode

To configure Arbitrary Waveform Generator in multi-instrument mode,

-   Python/MATLAB clients: `m.set_instrument(<slot>, ArbitraryWaveformGenerator)`, where `m` is the `MultiInstrument` object
-   cURL: `http://<ip>/api/<slot>/awg`

`<slot>` is required and depends on the `hardware` and `platform` combination. Read more about [slots](../../getting-started/starting-mim#selecting-the-multi-instrument-mode-configuration)

::: warning
Multi-instrument mode should be enabled before configuring Arbitrary Waveform Generator in one of the slots. Read [Platform](../moku/platform.md)
:::
