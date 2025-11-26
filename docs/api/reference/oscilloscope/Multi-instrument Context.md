---
title: Oscilloscope
prev: false
name: Multi-instrument
description: Oscilloscope in multi-instrument context
---

# Oscilloscope - Multi-Instrument Mode

To configure Oscilloscope in Multi-Instrument Mode,

-   Python/MATLAB clients: `m.set_instrument(<slot>, Oscilloscope)`, where `m` is the `MultiInstrument` object
-   cURL: `http://<ip>/api/<slot>/oscilloscope`

`<slot>` is required and depends on the `hardware` and `platform` combination. Read more about [slots](../../getting-started/starting-mim.md#selecting-the-multi-instrument-mode-configuration)

::: warning
Multi-Instrument Mode should be enabled before configuring Oscilloscope in one of the slots. Read [Platform](../moku/platform.md)
:::
