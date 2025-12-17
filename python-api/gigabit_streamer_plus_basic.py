# Gigabit Streamer+ Example
#
# This example demonstrates how to transmit and receive data using
# Moku Gigabit Streamer+ in single instrument mode.
#
# (c) Liquid Instruments Pty. Ltd.

from moku.instruments import GigabitStreamerPlus

# Connect to your Moku via its IP address. Set force_connect=True to
# overtake an existing connection.
i = GigabitStreamerPlus("192.168.XXX.XXX", force_connect=True)

try:
    # Set up the Gigabit Streamer+ inputs
    i.enable_input(channel=1, enable=True)
    i.enable_input(channel=2, enable=True)
    i.enable_input(channel=3, enable=True)
    i.enable_input(channel=4, enable=True)

    # Acquisition / interpolation
    i.set_acquisition(mode="Normal", sample_rate=1.25e9)
    i.set_interpolation(mode="Linear")

    # Analog frontend settings
    i.set_frontend(channel=1, impedance="50Ohm", coupling="DC", gain="0dB")
    i.set_frontend(channel=2, impedance="50Ohm", coupling="DC", gain="0dB")
    i.set_frontend(channel=3, impedance="50Ohm", coupling="DC", gain="0dB")
    i.set_frontend(channel=4, impedance="50Ohm", coupling="DC", gain="0dB")

    # Configure the local and remote network settings
    i.set_local_network(ip_address="10.10.1.1", port=5000)
    # Find host or configure the host IP address, UDP port, and MAC address
    i.set_remote_network(ip_address="168.192.XXX.XXX", port=5000, mac_address="A1:B2:C3:XX:XX:XX")
    i.set_outgoing_packets(mtu="1500bytes")

    # Enable outputs (channel, enable, gain, offset)
    i.set_output(channel=1, enable=True, gain=0.0, offset=0.0)
    i.set_output(channel=2, enable=True, gain=0.0, offset=0.0)
    i.set_output(channel=3, enable=True, gain=0.0, offset=0.0)
    i.set_output(channel=4, enable=True, gain=0.0, offset=0.0)

    # Print a summary of the instrument configuration
    print(i.summary())

    # Immediately start sending data for 10 seconds
    i.start_sending(duration=10)

except Exception as e:
    i.relinquish_ownership()
    raise e
finally:
    # Close the connection to the Moku device
    # This ensures network resources and released correctly
    i.relinquish_ownership()
