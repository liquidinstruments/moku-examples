---
title: Laser Lock Box
prev: false
name: Multi-instrument
description: Laser Lock Box in multi-instrument context
---

# Laser Lock Box - Multi-Instrument Mode

To configure LLB in Multi-Instrument Mode,

-   Python/MATLAB clients: `m.set_instrument(<slot>, LaserLockBox)`, where `m` is the `MultiInstrument` object
-   cURL: `http://<ip>/api/<slot>/laserlockbox`

`<slot>` is required and depends on the `hardware` and `platform` combination. Read more about [slots](../../getting-started/starting-mim.md#selecting-the-multi-instrument-mode-configuration)

::: warning
Multi-Instrument Mode should be enabled before configuring Laser Lock Box in one of the slots. Read [Platform](../moku/platform.md)
:::
