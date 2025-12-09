---
title: Lock-in Amplifier
prev: false
name: Multi-instrument
description: Lock-in Amplifier in multi-instrument context
---

# Lock-in Amplifier - Multi-Instrument Mode

To configure Lock-in Amplifier in Multi-Instrument Mode,

-   Python/MATLAB clients: `m.set_instrument(<slot>, LockInAmp)`, where `m` is the `MultiInstrument` object
-   cURL: `http://<ip>/api/<slot>/lockinamp`

`<slot>` is required and depends on the `hardware` and `platform` combination. Read more about [slots](../../getting-started/starting-mim.md#selecting-the-multi-instrument-mode-configuration)

::: warning
Multi-Instrument Mode should be enabled before configuring Lock-in Amplifier in one of the slots. Read [Platform](../moku/platform.md)
:::
