---
title: FIR Filter Builder
prev: false
name: Multi-instrument 
description: FIR Filter Builder in multi-instrument context
---

FIR Filter Builder - Multi-instrument mode
=====================================================

To configure FIR Filter Builder in multi-instrument mode, 

+ Python/MATLAB clients:  `m.set_instrument(<slot>, FIRFilterBuilder)`, where `m` is the `MultiInstrument` object
+ cURL:  `http://<ip>/api/<slot>/firfilter`

`<slot>` is required and depends on the `hardware` and `platform` combination. Read more about [slots](slots.md)

::: warning
Multi-instrument mode should be enabled before configuring FIR Filter Builder in one of the slots. Read [Platform](../moku/platform.md)
:::