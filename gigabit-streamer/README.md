# Chapter 1 - Welcome to Moku Cloud Compile (MCC) 

This project is intended to open the Moku to our users by allowing them to design and implement their own custom functionality. These custom designs can be deployed next to, and interact with the existing suite of Moku instruments using [Multi-instrument Mode](https://liquidinstruments.com/multi-instrument-mode/)(MiM). This allows you to prototype custom designs, interact with custom hardware or provide bespoke functions for your specific requirements.

This tutorial will guide you through the process to develop and deploy custom functions and features in minutes, not months, and without the overhead of complex software packages.  

```{note}
If you are already familiar with MCC, head right to <a href="https://compile.liquidinstruments.com/" target="_blank">Cloud Compile</a>.
```


## Unlocking the power of user-programmable FPGAs
Hardware description languages (HDL) like VHDL are notoriously difficult to master. However, they are essential for unlocking the computational power of field-programmable gate arrays (FPGAs) in digital design. Liquid Instruments aims to provide the power of HDL without much of the traditional burden associated with highly complex development environments.

### Why use FPGAs in test and measurement
**High-speed processing:**
FPGAs can handle high-speed data acquisition, processing, and analysis, enabling real-time testing and low-latency performance. With parallel processing, FPGAs execute multiple tasks simultaneously.

**Customization:**
Users can design and implement custom test algorithms, protocols, and signal processing on FPGAs to tailor the hardware to specific test requirements.

**Flexibility:**
FPGAs are highly flexible and can be reprogrammed to perform different tasks, a crucial benefit for test applications where requirements may evolve over time.

**Reliability:**
FPGAs are known for their robustness, a critical feature in applications where accuracy and repeatability are essential.

### Why use MCC
With MCC, you have the ability to add custom functionality to work in tandem with a powerful suite of test and measurement capabilities on the Moku:Go, Moku:Lab, and Moku:Pro. Additionally, Moku Cloud Compile aims to simplify FPGA programming and does not require the use of a complex development environment. Rather, MCC allows you to code, compile, and deploy directly from your browser.

```{figure} images/All3Moku.png
---
height: 500px
name: directive-fig
---
Moku suite of devices
```

Moku Cloud Compile is a powerful tool available exclusively for FPGA-based Moku devices from Liquid Instruments, allowing you to code, compile, and deploy custom algorithms.  These custom algorithms can be deployed alongside an expanding list of other software-defined test and measurement instruments, from bench essentials like an Oscilloscope and Spectrum Analyzer to advanced tools like a Lock-in Amplifier and Laser Lock Box. With MCC, it’s easy to create your own instruments, add functionality to our already extensive cabilities, build complex signal processing pipelines, or even test digital prototypes in conjunction with the existing embedded instruments.

## Multi-instrument and Slots

With the introduction of Multi-instrument Mode in Moku the FPGA has been divided into isolated regions we call 'slots'. Each slot can be configured with an instrument such as an Oscilloscope, Waveform Generator or your own MCC design which will run simultaneously and independendently. Slots can also be reconfigured to change their function without interrupting instruments running in other slots.

Each slot has several input and output ports which can be routed to or from various other locations. These sources can be the outputs of others slots or the physical ADCs of the Moku. Signals can be routed to other slots or to the DAC outputs of the Moku.

Multi-instrument Mode allows users of the Moku to build complete systems consisting of several instruments in flexible configurations to meet the signal processing requirements of their experiment. All of this is configurable using the Moku Application.

```{figure} images/multi_Instrument.png
---
height: 500px
name: directive-fig
---
Multi-instrument Mode on Moku:Pro
```

With the addition of the Moku Cloud Compiler, users can now include their own custom functionality in a multi-instrument configuration. This drastically increases the flexibility and utility of the Moku as an experimentation and system control platform.

### Slot Resources
The FPGA resources are divided between slots and the supporting logic in the platform surrounding the slots. The table below summarizes the resources availble to a custom design in each slot.

| | Moku:Pro (ZU9EG) | Moku:Go (ZC7020)	| Moku:Lab (ZC7020) |
|---|---|---|---|
| | 4 Slots | 2 Slots | 2 Slots |
|Core Clock	| 312.5MHz | 31.25MHz | 125MHz |
|LUT | 48400 | 20000 | 19600 |
|FF | 96800 | 40000 | 39200 |
|BRAM (36K) | 154 | 50 | 60 |
|DSP | 432 | 100 | 100 |

## FAQ
**How does MCC work?**
Moku Cloud Compile allows you to deploy custom DSP directly onto the Moku:Go, Moku:Lab, or Moku:Pro FPGA in Multi-instrument Mode. Write code using a web browser and compile it in the cloud; download and deploy the bitstream to your Moku device through the app.

**What is HDL?**
HDL, or hardware description language, refers to a family of programming languages used to describe digital logic circuits and program FPGAs. The most commonly used hardware description languages are VHDL and Verilog. We currently support VHDL, with Verilog support coming soon.

**How long does it take to compile?**
Although the total compilation time depends on the number of users engaged in the tool simultaneously, compilation can complete in as quickly as 15 minutes.

**What hardware platforms is Moku Cloud Compile compatible with?**
Currently Moku Cloud Compile is available on Moku:Pro, Moku:Lab, and Moku:Go running Multi-instrument Mode.

**Are there any examples to reference?**
Our team is constantly developing new content.  You are encouraged to visit the <a href="https://github.com/liquidinstruments/moku-examples" target="_blank">Liquid Instruments git repo</a> to review our extensive examples for MCC and Moku's enhanced capabilities with the APIs. 

We offer a range of resources to help you get started with Moku Cloud Compile. This tutorial is designed to get you started with Moku Cloud Compile and introduce you to key functionality and some advanced considerations and concepts.  You can explore additional resources and examples on our <a href="https://compile.liquidinstruments.com/docs/" target="_blank">MCC Documentation</a> page.

```{tableofcontents}
```
