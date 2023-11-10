---
title: PID Controller
prev: false
name: Multi-instrument 
description: PID Controller in multi-instrument context
---

PID Controller - Multi-instrument mode
=====================================================

To configure PID Controller in multi-instrument mode, 

+ Python/MATLAB clients:  `m.set_instrument(<slot>, PIDController)`, where `m` is the `MultiInstrument` object
+ cURL:  `http://<ip>/api/<slot>/pid`

`<slot>` is required and depends on the `hardware` and `platform` combination. Read more about [slots](../../starting-mim.md#selecting-the-multi-instrument-mode-configuration)

::: warning
Multi-instrument mode should be enabled before configuring PID Controller in one of the slots. Read [Platform](../moku/platform.md)
:::