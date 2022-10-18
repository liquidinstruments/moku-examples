---
title: Digital Filter Box
prev: false
name: Multi-instrument 
description: Digital Filter Box in multi-instrument context
---

Digital Filter Box - Multi-instrument mode
=====================================================

To configure Digital Filter Box in multi-instrument mode, 

+ Python/MATLAB clients:  `m.set_instrument(<slot>, DigitalFilterBox)`, where `m` is the `MultiInstrument` object
+ cURL:  `http://<ip>/api/<slot>/digitalfilterbox`

`<slot>` is required and depends on the `hardware` and `platform` combination. Read more about [slots](slots.md)

::: warning
Multi-instrument mode should be enabled before configuring Digital Filter Box in one of the slots. Read [Platform](../moku/platform.md)
:::