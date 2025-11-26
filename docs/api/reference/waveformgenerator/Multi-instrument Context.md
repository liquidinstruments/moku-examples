---
title: Waveform Generator
prev: false
name: Multi-instrument
description: Waveform Generator in multi-instrument context
---

# Waveform Generator - Multi-Instrument Mode

To configure Waveform Generator in Multi-Instrument Mode,

-   Python/MATLAB clients: `m.set_instrument(<slot>, WaveformGenerator)`, where `m` is the `MultiInstrument` object
-   cURL: `http://<ip>/api/<slot>/waveformgenerator`

`<slot>` is required and depends on the `hardware` and `platform` combination. Read more about [slots](../../getting-started/starting-mim.md#selecting-the-multi-instrument-mode-configuration)

::: warning
Multi-Instrument Mode should be enabled before configuring Waveform Generator in one of the slots. Read [Platform](../moku/platform.md)
:::
