#
# moku example: Gigabit Streamer
#
# This example demonstrates how you can stream on a Moku in loopback,
# receiving and transmitting data using Moku Gigabit Streamer, using
# the Oscilloscope to generate and view the data in real-time.
#
# Connect a SFP cable in loopback from SFP1 to SFP2, to allow
#
# |   Slot  1   |   Slot  2   |   Slot  3   |
# |     gs1     |     osc     |     gs2     |
# |     rx      |   gen and   |     tx      |
# |  from SFP1  | view signal |   to SFP2   |
#
#
# (c) Liquid Instruments Pty. Ltd.
#
import matplotlib.pyplot as plt

from moku.instruments import GigabitStreamer, MultiInstrument, Oscilloscope

# Connect to your Moku by its ip address MultiInstrument('192.168.###.###', platform_id=3)
# force_connect will overtake an existing connection
m = MultiInstrument("192.168.###.###", force_connect=True, platform_id=3)

try:
    # Set up Gigabit Streamer and Oscilloscope instruments
    gs1 = m.set_instrument(1, GigabitStreamer)  # Receiving Gigabit Streamer
    osc = m.set_instrument(2, Oscilloscope)  # Oscilloscope
    gs2 = m.set_instrument(3, GigabitStreamer)  # Transmitting Gigabit Streamer

    # Set up connections between the instruments
    # The Oscilloscope is connected to output to Gigabit Streamer in Slot 1 and read back
    # from Gigabit Streamer in Slot 3. Anytime a Gigabit Streamer instrument is connected
    # to Slot 1 it is transmitting and receiving data from SFP1.
    # Anytime a Gigabit streamer instrument is connected to Slot 2 it is transmitting
    # and receiving data from SFP2.
    connections = [
        dict(source="Slot1OutA", destination="Slot2InA"),
        dict(source="Slot2OutA", destination="Slot2InB"),
        dict(source="Slot2OutA", destination="Slot3InA"),
    ]
    m.set_connections(connections)

    # Configure the Oscilloscope to generate and view the signal in real-time
    # Generate a 1 MHz sine wave, set the trigger and timebase to view signal
    osc.generate_waveform(channel=1, type="Sine", amplitude=1, frequency=1e6, offset=0, phase=0)
    osc.set_trigger(type='Edge', source='ChannelA', level=0)
    osc.set_timebase(t1=-5e-6, t2=5e-6)

    # Set up the receiving Gigabit Streamer instrument
    #########################################################
    # Configure the acquisition settings
    gs1.enable_input(channel=1, enable=True)
    gs1.enable_input(channel=2, enable=True)
    gs1.set_acquisition(mode='Normal', sample_rate=156.25e6)
    gs1.set_interpolation(mode='Linear')

    # Configure the local and remote network settings
    gs1.set_local_network(ip_address="10.10.1.1", port=5000)
    # Get the MAC address of Gigabit Streamer 2 (connected to SFP2)
    gs2_mac = gs2.set_local_network(ip_address="10.10.1.2", port=5000)["mac_address"]
    gs1.set_remote_network(ip_address="10.10.1.2", port=5000, mac_address=gs2_mac)
    gs1.set_outgoing_packets(mtu="1500bytes")

    # Set up the transmitting Gigabit Streamer instrument
    #########################################################
    # Configure the acquisition settings
    gs2.enable_input(channel=1, enable=True)
    gs2.enable_input(channel=2, enable=False)
    gs2.set_acquisition(mode='Normal', sample_rate=156.25e6)
    gs2.set_interpolation(mode='Linear')

    # Configure the local and remote network settings
    gs2.set_local_network(ip_address="10.10.1.2", port=5000)
    # Get the MAC address of Gigabit Streamer 1 (connected to SFP1)
    gs1_mac = gs1.set_local_network(ip_address="10.10.1.1", port=5000)["mac_address"]
    gs2.set_remote_network(ip_address="10.10.1.1", port=5000, mac_address=gs1_mac)
    gs2.set_outgoing_packets(mtu="1500bytes")

    # Enable outputs and start sending data
    gs1.set_output(channel=1, enable=True, gain=0.0, offset=0.0)
    gs1.set_output(channel=2, enable=False, gain=0.0, offset=0.0)
    gs2.set_output(channel=1, enable=True, gain=0.0, offset=0.0)
    gs2.set_output(channel=2, enable=False, gain=0.0, offset=0.0)
    # Immediately start sending data for 10 seconds
    gs1.start_sending(duration=10, delay=0)
    gs2.start_sending(duration=10, delay=0)

    # Get initial data frame to set up plotting parameters. This can be done
    # once if we know that the axes aren't going to change (otherwise we'd do
    # this in the loop)
    data = osc.get_data()

    # Set up the plotting parameters
    plt.ion()
    plt.show()
    plt.grid(visible=True)
    plt.ylim([-1, 1])
    plt.xlim([data['time'][0], data['time'][-1]])

    (line1,) = plt.plot([])
    (line2,) = plt.plot([])

    # Configure labels for axes
    ax = plt.gca()

    # This loops continuously updates the plot with new data
    while True:
        # Get new data
        data = osc.get_data()

        # Update the plot
        line1.set_ydata(data['ch1'])  # Streamed data from Gigabit Streamer in Slot 1 (blue)
        line1.set_xdata(data['time'])
        line2.set_ydata(data['ch2'])  # Signal generated in Oscilloscope in loopback (orange)
        line2.set_xdata(data['time'])
        plt.pause(0.001)

except Exception as e:
    m.relinquish_ownership()
    raise e
finally:
    # Close the connection to the Moku device
    # This ensures network resources and released correctly
    m.relinquish_ownership()
