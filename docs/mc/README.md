---
tags:
    - Moku Compile
---

# Moku Compile

Welcome to Moku Compile!

This project is intended to open the Moku to our users by allowing them to design and implement their own custom functionality. These custom functions can be deployed next to, and interact with the existing suite of Moku instruments. This allows you to prototype custom designs, interact with custom hardware or provide bespoke functions for you specific requirements. 

### Customization with Multi-Instrument Mode

With the introduction of Multi-Instrument Mode in Moku the FPGA has
been divided into isolated regions we call 'slots'. Each slot can be configured
with an instrument such as an Oscilloscope or Waveform Generator which will run
simultaneously and independently. Multi-Instrument Mode allows users of the Moku to build complete systems consisting of several instruments in flexible configurations to meet the signal processing requirements of their experiment. All of this is configurable using the Moku application or API.

![An image](./multi-instrument.png)

With the addition of the Moku Compile, users can now include their own custom functionality in this multi-instrument configuration. The Moku Compile builds custom designs through Moku Cloud Compile that can be deployed as Custom Instrument on the Moku device.

These pages include references to help you start [building your custom designs](./getting-started/cloud.md) with information on the available FPGA resources on each [device](./slots.md). We also provide description to help you start coding using our [Custom Wrapper](./wrapper.md) or try creating a project with an example template.

<action-button text="Explore our example projects" link="https://github.com/liquidinstruments/moku-examples/tree/main/mcc" target="_blank"/>


<!-- ![An image](../public/cc-icon-solid.png) -->
