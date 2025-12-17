---
title: Gigabit Streamer
prev: false
name: Multi-Instrument
description: Gigabit Streamer in multi-instrument context
available_on: 'Moku:Delta'
---

# Gigabit Streamer - Multi-Instrument Mode

To configure **Gigabit Streamer** in Multi-Instrument Mode,

-   Python: `m.set_instrument(<slot>, GigabitStreamer)`, where `m` is the `MultiInstrument` object
-   MATLAB: `m.set_instrument(<slot>, @MokuGigabitStreamerPlus)`, where `m` is the `MultiInstrument` object
-   cURL: `http://<ip>/api/<slot>/gs`

To configure **Gigabit Streamer+** in Multi-Instrument Mode,

-   Python: `m.set_instrument(<slot>, GigabitStreamerPlus)`, where `m` is the `MultiInstrument` object
-   MATLAB: `m.set_instrument(<slot>, @MokuGigabitStreamerPlus)`, where `m` is the `MultiInstrument` object
-   cURL: `http://<ip>/api/<slot>/gbs`

`<slot>` is required and depends on the `hardware` and `platform` combination. Read more about [slots](../../getting-started/starting-mim.md#selecting-the-multi-instrument-mode-configuration)

::: warning
Multi-Instrument Mode should be enabled before configuring Gigabit Streamer in one of the slots. Read [Platform](../moku/platform.md)
:::
