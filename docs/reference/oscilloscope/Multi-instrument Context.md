---
title: Oscilloscope
prev: false
name: Multi-instrument 
description: Oscilloscope in multi-instrument context
---

Oscilloscope - Multi-instrument mode
=====================================================

To configure Oscilloscope in multi-instrument mode, 

+ Python/MATLAB clients:  `m.set_instrument(<slot>, Oscilloscope)`, where `m` is the `MultiInstrument` object
+ cURL:  `http://<ip>/api/<slot>/oscilloscope`

`<slot>` is required and depends on the `hardware` and `platform` combination. Read more about [slots](slots.md)

::: warning
Multi-instrument mode should be enabled before configuring Oscilloscope in one of the slots. Read [Platform](../moku/platform.md)
:::