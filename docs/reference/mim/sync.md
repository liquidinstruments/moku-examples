---
additional_doc: Calling this function is the equivalent of pressing the "sync" button in the UI
description: Synchronize the phase of internal oscillators and across all the instrument slots
method: get
name: sync
parameters: []
summary: sync
---

<headers/>

When running multiple instruments simultaneously, it is often required that one or more of them have a common concept of phase. For example, a Waveform Generator generating a modulation signal, and a Lock-in Amplifier demodulating it, should share the concept of "zero phase".

This is accomplished by configuring each instrument individually, then calling this `sync` function to simultaneously reset all phase counters.

<parameters/>
