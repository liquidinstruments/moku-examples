---
title: Lock-in Amplifier
prev: false
name: Multi-instrument 
description: Lock-in Amplifier in multi-instrument context
---

Lock-in Amplifier - Multi-instrument mode
=====================================================

To configure Lock-in Amplifier in multi-instrument mode, 

+ Python/MATLAB clients:  `m.set_instrument(<slot>, LockInAmp)`, where `m` is the `MultiInstrument` object
+ cURL:  `http://<ip>/api/<slot>/lockinamp`

`<slot>` is required and depends on the `hardware` and `platform` combination. Read more about [slots](../../starting-mim.md#selecting-the-multi-instrument-mode-configuration)

::: warning
Multi-instrument mode should be enabled before configuring Lock-in Amplifier in one of the slots. Read [Platform](../moku/platform.md)
:::