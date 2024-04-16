---
title: Time & Frequency Analyzer
prev: false
name: Multi-instrument 
description: Time & Frequency Analyzer in multi-instrument context
---

Time & Frequency Analyzer - Multi-instrument mode
=====================================================

To configure Time & Frequency Analyzer in multi-instrument mode, 

+ Python/MATLAB clients:  `m.set_instrument(<slot>, TimeFrequencyAnalyzer)`, where `m` is the `MultiInstrument` object
+ cURL:  `http://<ip>/api/<slot>/tfa`

`<slot>` is required and depends on the `hardware` and `platform` combination. Read more about [slots](../../starting-mim.md#selecting-the-multi-instrument-mode-configuration)

::: warning
Multi-instrument mode should be enabled before configuring Spectrum Analyzer in one of the slots. Read [Platform](../moku/platform.md)
:::