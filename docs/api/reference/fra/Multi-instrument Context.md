---
title: Frequency Response Analyzer
prev: false
name: Multi-instrument
description: Frequency Response Analyzer in multi-instrument context
---

# Frequency Response Analyzer - Multi-Instrument Mode

To configure Frequency Response Analyzer in Multi-Instrument Mode,

-   Python/MATLAB clients: `m.set_instrument(<slot>, FrequencyResponseAnalyzer)`, where `m` is the `MultiInstrument` object
-   cURL: `http://<ip>/api/<slot>/fra`

`<slot>` is required and depends on the `hardware` and `platform` combination. Read more about [slots](../../getting-started/starting-mim.md#selecting-the-multi-instrument-mode-configuration)

::: warning
Multi-Instrument Mode should be enabled before configuring Frequency Response Analyzer in one of the slots. Read [Platform](../moku/platform.md)
:::
