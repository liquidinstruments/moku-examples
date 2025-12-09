---
title: Phasemeter
prev: false
name: Multi-instrument
description: Phasemeter in multi-instrument context
---

# Phasemeter - Multi-Instrument Mode

To configure Phasemeter in Multi-Instrument Mode,

-   Python/MATLAB clients: `m.set_instrument(<slot>, Phasemeter)`, where `m` is the `MultiInstrument` object
-   cURL: `http://<ip>/api/<slot>/phasemeter`

`<slot>` is required and depends on the `hardware` and `platform` combination. Read more about [slots](../../getting-started/starting-mim.md#selecting-the-multi-instrument-mode-configuration)

::: warning
Multi-Instrument Mode should be enabled before configuring Phasemeter in one of the slots. Read [Platform](../moku/platform.md)
:::
