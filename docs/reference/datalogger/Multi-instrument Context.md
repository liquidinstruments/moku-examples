---
title: Datalogger
prev: false
name: Multi-instrument 
description: Datalogger in multi-instrument context
---

Datalogger - Multi-instrument mode
=====================================================

To configure Datalogger in multi-instrument mode, 

+ Python/MATLAB clients:  `m.set_instrument(<slot>, Datalogger)`, where `m` is the `MultiInstrument` object
+ cURL:  `http://<ip>/api/<slot>/datalogger`

`<slot>` is required and depends on the `hardware` and `platform` combination. Read more about [slots](slots.md)

::: warning
Multi-instrument mode should be enabled before configuring Datalogger in one of the slots. Read [Platform](../moku/platform.md)
:::