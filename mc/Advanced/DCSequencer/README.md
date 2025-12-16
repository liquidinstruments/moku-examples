# DC Sequencer

A customer required a system that output a series of DC values, stepping between them based on a trigger input. Some AWG units can segment their memory and transition between segments in this way, but it is a complicated process given each segment contains a single value.

Using Moku Compile, a simple sequencer was written with custom schmitt trigger logic and a pre-defined list of DAC output values. The debounced trigger was output from the logic as well, to be used downstream.

The schmitt trigger levels are configurable through a Control register, the DC value sequence is hard-coded.
