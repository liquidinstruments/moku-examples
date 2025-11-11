---
title: Neural Network
prev: false
name: Multi-instrument
description: Neural Network in multi-instrument context
---

# Neural Network - Multi-instrument mode

To configure the Neural Network in multi-instrument mode,

-   Python/MATLAB clients: `m.set_instrument(<slot>, NeuralNetwork)`, where `m` is the `MultiInstrument` object
-   cURL: `http://<ip>/api/<slot>/neuralnetwork`

`<slot>` is required and depends on the `hardware` and `platform` combination. Read more about [slots](../../getting-started/starting-mim.md#selecting-the-multi-instrument-mode-configuration)

::: warning
Multi-instrument mode should be enabled before configuring Neural Network in one of the slots. Read [Platform](../moku/platform.md)
:::
