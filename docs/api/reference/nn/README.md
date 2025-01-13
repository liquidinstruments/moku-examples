---
title: Neural Network
prev: false
available_on: 'Moku:Pro'
---

# Neural Network

The Neural Network enables the deployment of real-time deep learning algorithms, in a .linn format, into your Multi-Instrument setup. See the [examples](../../../mnn/examples) and [documentation](../../../mnn/linnmodel-class/linnmodel.md) for how to write and export your neural network .linn file in Python.

If you are directly using the RESTful API (e.g. using cURL), the instrument name as used in the URL is `neuralnetwork`.

:::warning Multi-instrument Mode
Moku Neural Network instruments can only be used in Multi-instrument Mode. Refer to the Multi-instrument Mode [Getting Started Guide](../../getting-started/starting-mim.md) for more details.
:::

:::tip Loading your .linn file
Before you can deploy a Moku Neural Network instrument, you must load the .linn file that you have generated in Python onto your Moku. When using the API, this can be accomplished through the [upload_network](./upload_network.md) function.
:::

<function-index/>
