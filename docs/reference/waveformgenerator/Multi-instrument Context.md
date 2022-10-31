---
title: Waveform Generator
prev: false
name: Multi-instrument 
description: Waveform Generator in multi-instrument context
---

Waveform Generator - Multi-instrument mode
=====================================================

To configure Waveform Generator in multi-instrument mode, 

+ Python/MATLAB clients:  `m.set_instrument(<slot>, WaveformGenerator)`, where `m` is the `MultiInstrument` object
+ cURL:  `http://<ip>/api/<slot>/waveformgenerator`

`<slot>` is required and depends on the `hardware` and `platform` combination. Read more about [slots](slots.md)

::: warning
Multi-instrument mode should be enabled before configuring Waveform Generator in one of the slots. Read [Platform](../moku/platform.md)
:::