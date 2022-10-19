---
title: Cloud Compile
prev: false
name: Multi-instrument 
description: Cloud Compile in multi-instrument context
---

Cloud Compile - Multi-instrument mode
=====================================================

To configure Cloud Compile in multi-instrument mode, 

+ Python/MATLAB clients:  `m.set_instrument(<slot>, CloudCompile)`, where `m` is the `MultiInstrument` object
+ cURL:  `http://<ip>/api/<slot>/cloudcompile`

`<slot>` is required and depends on the `hardware` and `platform` combination. Read more about [slots](slots.md)

::: warning
Multi-instrument mode should be enabled before configuring Cloud Compile in one of the slots. Read [Platform](../moku/platform.md)
:::

::: tip
Cloud Compile Instrument can only be used with Multi-instrument mode enabled
:::