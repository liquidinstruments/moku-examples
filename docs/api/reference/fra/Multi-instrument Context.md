---
title: Frequency Response Analyzer
prev: false
name: Multi-instrument
description: Frequency Response Analyzer in multi-instrument context
---

# Frequency Response Analyzer - Multi-instrument mode

To configure Frequency Response Analyzer in multi-instrument mode,

-   Python/MATLAB clients: `m.set_instrument(<slot>, FrequencyResponseAnalyzer)`, where `m` is the `MultiInstrument` object
-   cURL: `http://<ip>/api/<slot>/fra`

`<slot>` is required and depends on the `hardware` and `platform` combination. Read more about [slots](../../getting-started/starting-mim#selecting-the-multi-instrument-mode-configuration)

::: warning
Multi-instrument mode should be enabled before configuring Frequency Response Analyzer in one of the slots. Read [Platform](../moku/platform)
:::
