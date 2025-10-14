#
# moku example: Basic Logic Analyzer
#
# This example demonstrates how to use Pattern Generator in
# Logic Analyzer and generate and observe patterns on DIO pins.
#
# (c) Liquid Instruments Pty. Ltd.
#

import matplotlib.pyplot as plt

from moku.instruments import LogicAnalyzer

# Connect to your Moku by its ip address using
# LogicAnalyzer('192.168.###.###')
# force_connect will overtake an existing connection
i = LogicAnalyzer('192.168.###.###', force_connect=True)

try:
    patterns = [
        {"pin": 1, "pattern": [1] * 1024},  # logic high
        {"pin": 2, "pattern": [0] * 1024},  # logic low
        {"pin": 3, "pattern": [0, 1] * 512},
        {"pin": 4, "pattern": [1, 0] * 512},
    ]

    i.set_pattern_generator(1, patterns=patterns, divider=8)

    i.set_pin_mode(pin=1, state="PG1")
    i.set_pin_mode(pin=2, state="PG1")
    i.set_pin_mode(pin=3, state="PG1")
    i.set_pin_mode(pin=4, state="PG1")
    data = i.get_data(wait_reacquire=True, include_pins=[1, 2, 3, 4])

    (pin1,) = plt.step(data["time"], data["pin1"])
    (pin2,) = plt.step(data["time"], [i + 2 for i in data["pin2"]])
    (pin3,) = plt.step(data["time"], [i + 4 for i in data["pin3"]])
    (pin4,) = plt.step(data["time"], [i + 6 for i in data["pin4"]])

    plt.ion()
    plt.show()
    plt.grid(True)
    plt.ylim([-1, 8])
    plt.yticks([0, 2, 4, 6], labels=["Pin1", "Pin2", "Pin3", "Pin4"])

    while True:
        data = i.get_data(wait_reacquire=True, include_pins=[1, 2, 3, 4])

        pin1.set_xdata(data["time"])
        pin2.set_xdata(data["time"])
        pin3.set_xdata(data["time"])
        pin4.set_xdata(data["time"])

        pin1.set_ydata(data["pin1"])
        pin2.set_ydata([i + 2 for i in data["pin2"]])
        pin3.set_ydata([i + 4 for i in data["pin3"]])
        pin4.set_ydata([i + 6 for i in data["pin4"]])

        plt.pause(0.001)


except Exception as e:
    i.relinquish_ownership()
    raise e
finally:
    # Close the connection to the Moku device
    # This ensures network resources and released correctly
    i.relinquish_ownership()
