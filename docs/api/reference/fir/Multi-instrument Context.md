---
title: FIR Filter Builder
prev: false
name: Multi-instrument
description: FIR Filter Builder in multi-instrument context
---

# FIR Filter Builder - Multi-Instrument Mode

To configure FIR Filter Builder in Multi-Instrument Mode,

-   Python/MATLAB clients: `m.set_instrument(<slot>, FIRFilterBuilder)`, where `m` is the `MultiInstrument` object
-   cURL: `http://<ip>/api/<slot>/firfilter`

`<slot>` is required and depends on the `hardware` and `platform` combination. Read more about [slots](../../getting-started/starting-mim.md#selecting-the-multi-instrument-mode-configuration)

::: warning
Multi-Instrument Mode should be enabled before configuring FIR Filter Builder in one of the slots. Read [Platform](../moku/platform.md)
:::
