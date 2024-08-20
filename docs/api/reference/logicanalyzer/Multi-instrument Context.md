---
title: Logic Analyzer
prev: false
name: Multi-instrument
description: Logic Analyzer in multi-instrument context
---

# Logic Analyzer - Multi-instrument mode

To configure Logic Analyzer in multi-instrument mode,

-   Python/MATLAB clients: `m.set_instrument(<slot>, LogicAnalyzer)`, where `m` is the `MultiInstrument` object
-   cURL: `http://<ip>/api/<slot>/lockinamp`

`<slot>` is required and depends on the `hardware` and `platform` combination. Read more about [slots](../../getting-started/starting-mim.md#selecting-the-multi-instrument-mode-configuration)

::: warning
Multi-instrument mode should be enabled before configuring Lock-in Amplifier in one of the slots. Read [Platform](../moku/platform.md)
:::
