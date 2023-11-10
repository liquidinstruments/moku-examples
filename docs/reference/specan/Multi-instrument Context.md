---
title: Spectrum Analyzer
prev: false
name: Multi-instrument 
description: Spectrum Analyzer in multi-instrument context
---

Spectrum Analyzer - Multi-instrument mode
=====================================================

To configure Spectrum Analyzer in multi-instrument mode, 

+ Python/MATLAB clients:  `m.set_instrument(<slot>, SpectrumAnalyzer)`, where `m` is the `MultiInstrument` object
+ cURL:  `http://<ip>/api/<slot>/spectrumanalyzer`

`<slot>` is required and depends on the `hardware` and `platform` combination. Read more about [slots](../../starting-mim.md#selecting-the-multi-instrument-mode-configuration)

::: warning
Multi-instrument mode should be enabled before configuring Spectrum Analyzer in one of the slots. Read [Platform](../moku/platform.md)
:::