---
title: Laser Lock Box
prev: false
name: Multi-instrument 
description: Laser Lock Box in multi-instrument context
---

Laser Lock Box - Multi-instrument mode
=====================================================

To configure LLB in multi-instrument mode, 

+ Python/MATLAB clients:  `m.set_instrument(<slot>, LaserLockBox)`, where `m` is the `MultiInstrument` object
+ cURL:  `http://<ip>/api/<slot>/laserlockbox`

`<slot>` is required and depends on the `hardware` and `platform` combination. Read more about [slots](slots.md)

::: warning
Multi-instrument mode should be enabled before configuring Laser Lock Box in one of the slots. Read [Platform](../moku/platform.md)
:::