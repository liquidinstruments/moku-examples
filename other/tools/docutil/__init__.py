# Define map between spec path and the instrument directory name,
# this essentially compares the methods containing path with
# methods defined in that particular directory
INSTRUMENTS_MAP = dict(phasemeter="phasemeter", fra="fra",
                       logicanalyzer="logicanalyzer", lockinamp="lia",
                       waveformgenerator="waveformgenerator",
                       spectrumanalyzer="specan",
                       oscilloscope="oscilloscope",
                       datalogger="datalogger", firfilter="fir",
                       digitalfilterbox="dfb",
                       pidcontroller="pid", awg="awg")

NON_INSTRUMENTS_MAP = dict(mim="mim", moku="moku")
