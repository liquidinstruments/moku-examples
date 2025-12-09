---
title: Datalogger
prev: false
name: Multi-instrument
description: Datalogger in multi-instrument context
---

# Datalogger - Multi-Instrument Mode

To configure Datalogger in Multi-Instrument Mode,

-   Python/MATLAB clients: `m.set_instrument(<slot>, Datalogger)`, where `m` is the `MultiInstrument` object
-   cURL: `http://<ip>/api/<slot>/datalogger`

`<slot>` is required and depends on the `hardware` and `platform` combination. Read more about [slots](../../getting-started/starting-mim.md#selecting-the-multi-instrument-mode-configuration)

::: warning
Multi-Instrument Mode should be enabled before configuring Datalogger in one of the slots. Read [Platform](../moku/platform.md)
:::
