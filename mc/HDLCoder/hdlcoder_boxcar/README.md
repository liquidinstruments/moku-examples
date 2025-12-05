# Operation Instructions

This repository is associated with the [Boxcar Averager](https://liquidinstruments.com/application-notes/snr-boxcar-averager/) application note.

## Python Control Panel Guidelines

### Download Control Panel and Bitstreams

- The Python control panel can be found in the [Python](./python/) folder.
- The bitstreams can be found in the [bitstreams](https://liquidinstruments.com/application-notes/building-dual-boxcar-averagers/) folder.

### Install Moku-Python package

Please refer to the [Moku-Python API reference page](https://apis.liquidinstruments.com/starting-python.html) for installation instructions.

If this is your first time installing Moku API package, pleas also install `setuptools` by running the following command in the command line window:

```
pip install setuptools
```

### Install Additional Packages

The Python script requires `matplotlib` for plot the figures. Please install it by running the following command in the command line window:

```
pip install matplotlib
```

### Modify the Python code

To control your device with the Python control panel, you may need to modify a few lines in the script.

#### 1. Change the IP Address and Bitstreams Path

Each device has a unique IP address, and the bitstreams path may vary depending on your system. Please modify `1` and `2` in the Python script to reflect your setup:

```
IP_ADDR = '192.168.2.200'
BITSTREAMS_PATH = 'C:/Users/heyan/Downloads/boxcarMokuPro.tar.gz'
```

#### 2. Update Analog Frontend Settings
- Moku:Lab, Moku:Pro and Moku:Delta support both `50Ohm` and `1MOhm` input impedances, while Moku:Go only supports `1MOhm`.
- For additional forntend settings, please refer to the [`set_frontend`](https://apis.liquidinstruments.com/reference/mim/set_frontend.html) API documentation. 

Please modify line `3 to 5` as needed:

```
ATTENUATION = '0dB'
INPUT_IMPEDANCE = '50Ohm'
COUPLING = 'DC'
```

Additionally, Moku:Pro uses `platform_id` of `4`, Moku:Delta uses `3` or `8` and Moku:Go and Moku:Lab `2` or `3`. For more details, please visit [Multi-instrument Mode API](https://apis.liquidinstruments.com/starting-mim.html).

```
m = MultiInstrument(IP_ADDR, platform_id=4, force_connect=True)
```

#### 3. Change initial settings
There are a few boxcar averager configuration options. For example, trigger threshold, boxcar gate width, and etc. It is necessary to update these lines if you wish to run directly:

```
INITIAL_NEGATIVE_TIME = -200 #ns
INITIAL_POSITIVE_TIME = 800 #ns
INITIAL_TRIGGER_LEVEL = 0.075 #Volts
INITIAL_TRIGGER_DELAY = 1440 #ns
INITIAL_GATEWIDTH = 320 #ns
INITIAL_AVERAGE_LENGTH = 100
INITIAL_OUTPUT_GAIN = 1e-2
```

If you have any questions, please feel free to post your questions on the [Liquid Instruments Support Forum](https://forum.liquidinstruments.com/).

## Compile Bitstreams

The boxcar averager VHDL codes can be found in the [`vhdl_codes`](./vhdl_codes/) folder. To compile your own bitstreams, please follow the [Moku Compile Reference](https://apis.liquidinstruments.com/mc/deploying.html#building) to build the bitstreams from the provided VHDL codes.

For more information about Moku Compile, please refer to the [Moku Compile Getting Started Guide](https://liquidinstruments.com/application-notes/moku-cloud-compile-getting-started-guide).

## Modify the Design

Modifying the VHDL codes directly can be challenging. The origianl MathWorks Simulink model is attached as [`Boxcar Simulink`](./vhdl_codes/BoxcarAveragerFixedPoint.slx). You can change the model and generate your own VHDL codes, then create your custom bitstreams. For more information on using MathWorks HDL Coder, please refer to the [application note](https://liquidinstruments.com/application-notes/cloud-compile-with-mathworks-hdl-coder-pt-2/).