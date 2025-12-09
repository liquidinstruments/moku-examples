---
title: Neural Network
prev: false
name: Multi-instrument
description: Neural Network in multi-instrument context
---

# Neural Network - Multi-Instrument Mode

To configure the Neural Network in Multi-Instrument Mode,

-   Python/MATLAB clients: `m.set_instrument(<slot>, NeuralNetwork)`, where `m` is the `MultiInstrument` object
-   cURL: `http://<ip>/api/<slot>/neuralnetwork`

`<slot>` is required and depends on the `hardware` and `platform` combination. Read more about [slots](../../getting-started/starting-mim.md#selecting-the-multi-instrument-mode-configuration)

::: warning
Multi-Instrument Mode should be enabled before configuring Neural Network in one of the slots. Read [Platform](../moku/platform.md)
:::
