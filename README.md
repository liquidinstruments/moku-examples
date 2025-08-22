# Moku Examples - Curated Edition

This is a curated subset of the [upstream moku-examples repository](https://github.com/liquidinstruments/moku-examples) maintained by [Sealablab](https://github.com/sealablab).

## 🎯 **What's Included**

### **Python API Examples** (4.0MB)
- Moku API usage examples
- Cloud Compile examples
- Jupyter notebooks for various instruments
- Basic to advanced Python scripts

### **MCC (VHDL/Verilog) Examples** (23MB)
- Hardware design examples for Moku Cloud Compile
- Basic building blocks (Adder, ClockDivider, DIO, DSP)
- Advanced examples (EventCounter, VGA Display, Servo control)
- IP Core templates and examples

## ❌ **What Was Removed**

This curated edition removes the following to focus on core development needs:
- **Neural Network examples** (292MB) - Large notebooks with embedded data
- **MATLAB API examples** (128KB) - Not relevant for Python/VHDL development
- **Other language APIs** (44KB) - Not relevant for current development

## 🚀 **Getting Started**

### **For Python Development**
- Browse the `python-api/` directory for Moku API examples
- Start with basic examples like `hello_moku.ipynb`

### **For VHDL/Verilog Development**
- Explore the `mcc/` directory for hardware examples
- Begin with basic examples in `mcc/Basic/`

## 📚 **Repository Structure**

```
moku-examples-clean/
├── python-api/          # Python examples and notebooks
├── mcc/                 # VHDL/Verilog hardware examples
├── LICENSE              # MIT license
├── .gitignore          # Git configuration
└── README.md           # This file
```

## 🔗 **Related Repositories**

- **Main Workspace**: [moku-vhdl-dev-workspace](https://github.com/sealablab/moku-vhdl-dev-workspace)
- **Python Tools**: [moku-dev-python](https://github.com/sealablab/moku-dev-python)
- **VHDL Development**: [moku-dev-vhdl](https://github.com/sealablab/moku-dev-vhdl)

## 📄 **License**

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 **Contributing**

This is a curated fork. For contributions to the main examples repository, please visit [liquidinstruments/moku-examples](https://github.com/liquidinstruments/moku-examples).
