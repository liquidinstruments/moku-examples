---
title: Phasemeter
prev: false
name: Multi-instrument 
description: Phasemeter in multi-instrument context
---

Phasemeter - Multi-instrument mode
=====================================================

To configure Phasemeter in multi-instrument mode, 

+ Python/MATLAB clients:  `m.set_instrument(<slot>, Phasemeter)`, where `m` is the `MultiInstrument` object
+ cURL:  `http://<ip>/api/<slot>/phasemeter`

`<slot>` is required and depends on the `hardware` and `platform` combination. Read more about [slots](slots.md)

::: warning
Multi-instrument mode should be enabled before configuring Phasemeter in one of the slots. Read [Platform](../moku/platform.md)
:::