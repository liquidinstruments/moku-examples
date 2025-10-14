# Moku Examples

A comprehensive collection of examples demonstrating how to use [Liquid Instruments Moku devices](https://www.liquidinstruments.com/) across multiple programming languages and environments.

## Overview

This repository contains example code for working with Moku devices, including:
- **Python API examples** - Scripts and Jupyter notebooks for Python-based instrument control
- **MATLAB API examples** - MATLAB scripts for instrument automation
- **Neural Network examples** - Deep learning applications using the Moku Neural Network instrument
- **Multi-language examples** - Demonstrations in other programming languages

## Prerequisites

### Hardware
- A Moku device
- Network connection to your Moku device

### Software
- **For Python examples**: Python 3.9 or higher
- **For MATLAB examples**: MATLAB R2016b or later

## Important: Configuration Required

**Before running any example**, you must edit the file to configure your device connection:
- Update the IP address (e.g., `'192.168.1.100'`) or device name to match your Moku device
- Some examples may require additional configuration (sample rates, channels, etc.)

Look for lines like:
```python
MOKU_IP = '192.168.1.100'  # Update with your device IP
# or
i = Oscilloscope('192.168.###.###', force_connect=True)
```

**No example will work "out of the box"** without this configuration step.

## Quick Start

### Python Examples

1. **Install the Moku Python API:**
   ```bash
   pip install moku
   ```

2. **Run a basic example:**
   ```bash
   python python-api/oscilloscope_plotting.py
   ```

3. **Try a Jupyter notebook:**
   ```bash
   jupyter notebook python-api/hello_moku.ipynb
   ```

### MATLAB Examples

1. **Install the Moku MATLAB API** from the [Liquid Instruments website](https://www.liquidinstruments.com/downloads/)

2. **Run an example:**
   ```matlab
   run('matlab-api/oscilloscope_basic.m')
   ```

### Neural Network Examples

1. **Install neural network dependencies:**
   ```bash
   pip install 'moku[neuralnetwork]'
   ```

2. **Run a neural network example:**
   ```bash
   jupyter notebook neural-network/Simple_sine.ipynb
   ```

## Repository Structure

```
moku-examples/
├── python-api/          # Python scripts and Jupyter notebooks
│   ├── oscilloscope_*.py
│   ├── datalogger_*.py
│   ├── waveformgenerator_*.py
│   └── ...
├── matlab-api/          # MATLAB scripts (.m files)
│   ├── oscilloscope_*.m
│   ├── laser_lock_box_*.m
│   └── ...
├── neural-network/      # Neural network instrument examples
│   ├── Classification.ipynb
│   ├── Autoencoder.ipynb
│   └── ...
├── other-language-api/  # Examples in other languages
└── mcc/                 # Multi-instrument Cloud Compile examples
```

## Example Categories

### Python API
The `python-api/` directory contains examples for:
- **Oscilloscope**: Data acquisition, plotting, streaming, deep memory mode
- **Waveform Generator**: Signal generation, modulation, triggering
- **Data Logger**: High-speed data logging and streaming
- **Spectrum Analyzer**: Frequency domain analysis
- **Lock-in Amplifier**: Phase-sensitive detection
- **Phasemeter**: Phase and frequency measurements
- **PID Controller**: Feedback control systems
- **Logic Analyzer**: Digital signal analysis and protocol decoding
- **FIR Filter**: Digital signal filtering
- **Laser Lock Box**: Laser stabilization
- **Multi-Instrument Mode (MIM)**: Using multiple instruments simultaneously

### Neural Network
The `neural-network/` directory contains examples for:
- **Signal Classification**: Detecting anomalies in time-series data
- **Autoencoders**: Signal compression and reconstruction
- **Regression**: Predicting signal characteristics
- **Control Systems**: Neural network-based PID control
- **Signal Processing**: FFT, filtering, and signal identification

### MATLAB API
The `matlab-api/` directory mirrors the Python examples with MATLAB implementations.

## Finding the Right Example

### By Instrument Type
- **Oscilloscope**: `*oscilloscope*` files
- **Waveform Generator**: `*waveformgenerator*` or `*wavegen*` files
- **Data Logger**: `*datalogger*` files
- **Spectrum Analyzer**: `*spectrumanalyzer*` files
- **And more...**

### By Feature
- **Plotting**: `*plotting*` files - Show how to visualize data
- **Streaming**: `*streaming*` files - Real-time data streaming
- **Basic**: `*basic*` files - Simple getting-started examples
- **MIM**: `mim_*` files - Multi-instrument mode examples

## Documentation

- [Moku Python API Documentation](https://apis.liquidinstruments.com/python/)
- [Moku MATLAB API Documentation](https://apis.liquidinstruments.com/matlab/)
- [Moku User Manuals](https://www.liquidinstruments.com/resources/)
- [Neural Network Instrument Guide](https://www.liquidinstruments.com/neural-network/)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Setting up your development environment
- Running tests and code quality checks
- Submitting pull requests

## Support

- **Issues**: Report bugs or request examples via [GitHub Issues](https://github.com/liquidinstruments/moku-examples/issues)
- **Email**: support@liquidinstruments.com
- **Forum**: [Liquid Instruments Community Forum](https://forum.liquidinstruments.com/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Examples and code maintained by [Liquid Instruments](https://www.liquidinstruments.com/).
