## Moku Neural Network

The **Moku Neural Network** enables the deployment of real-time deep learning algorithms directly within your test instruments. Unlike traditional machine learning models that require extensive training and specialized knowledge, the Moku Neural Network offers a flexible, FPGA-based architecture designed for low-latency inference. This allows for efficient, real-time processing without the long training times typically associated with neural networks.

The Moku Neural Network is optimized for tasks like:

-   Closed-loop control
-   Noise filtering
-   Signal classification
-   Accuracy detection
-   Quadrant sensor control

## Neural Network Applications

Here are some practical applications where the Moku Neural Network can be used:

-   **Signal Classification**: Helps in identifying patterns in noisy signals, making real-time data processing more accurate and reliable.
-   **Quadrant Photodiode Sensing**: Enhances your photodiode-based systems for accurate light detection, essential for various optical setups.
-   **Building a Neural Network**: You can construct a neural network using Python, and then deploy it directly to the Moku platform.

## Getting Started

### 1. Requirements

-   [Python](https://www.python.org) >= **3.9**.
-   Your Moku connected to the same network as your computer.
-   Internet access.

### 2. Check your Python Installation

At a command prompt (e.g. cmd.exe, Windows Terminal, MacOS Terminal) check your Python version. It should be greater than or equal to 3.9.0.

```
$ python --version
Python 3.9.0
```

### 3. upgrade or install the `moku` Library and install Neural Network dependencies

It is recommended to do your Moku Neural Network development in a virtual environment, see Python's guide to [installing virtual environments](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/). Once you have activated your project's venv, continue installing the `moku` Library.

**To upgrade** your installed `moku` Library enter the following command in a terminal.

```
$ pip install --upgrade moku
```

**To install** the `moku` library, we will use `pip` in a command prompt terminal. You can easily check that the installation succeeded by running the simple Python command listed below. If there is no output from the Python command, then the installation has succeeded. If you see an error message, refer to [Troubleshooting](https://apis.liquidinstruments.com/api/getting-started/starting-python.html#troubleshooting).

```
$ pip install moku
$ python -c 'import moku'
```

Then install the Moku Neural Network instrument and it's machine learning dependencies with:

```
$ pip install 'moku[neuralnetwork]'
```

### 4. Find Your IP Address

The IP address of your Moku: device can be found with

```
$ moku list
Name                 Serial  HW     FW     IP                  
---------------------------------------------------------------
MokuPro-001234        1234    Pro    600    fe80::94db:946e:8d4e:129e
```

### 5. Install Python dependencies for examples

-   Numpy
-   Ipykernel
-   Matplotlib
-   Tqdm
-   SciPy

These dependences are to run our [Examples](./examples/), but are not needed to build a network. Check that each dependency is installed by running `pip list` in your terminal to view all installed packages. Or install with:

```
$ pip install numpy ipykernel matplotlib tqdm scipy
```

### 6. Start scripting

You are now ready to start scripting your own neural network. Check out our [Examples](./examples/) for more inspiration. Happy Coding!

## Moku Device and Slots

The Neural Network can be implemented on all Moku devices with the same number of layers maintained across the devices. However, the number of neurons and the maximum sampling rates are different depending on the device and the Multi-Instrument Mode slots. These act as the constraints when designing the Linn model. Using a Linn model on a device with neurons beyond its limit cannot be uploaded into the instrument. The table below summarizes these parameters on different configurations


|                             | Moku:Delta  | Moku:Delta |   Moku:Pro | Moku:Lab | Moku:Lab |    Moku:Go |    Moku:Go | 
| --------------------------- | ----------: | ---------: | ---------: | -------: | -------: |  --------: |  --------: | 
|             Number of Slots |     3 Slots |    8 Slots |    4 Slots |  2 Slots |  3 Slots |    2 Slots |    3 Slots | 
| Maximum input sampling rate |    305kSa/s |   305kSa/s |   305kSa/s | 122kSa/s | 122kSa/s | 30.5 kSa/s | 30.5 kSa/s | 
|   Maximum neurons per layer |         100 |        100 |        100 |       80 |       50 |         80 |         50 | 