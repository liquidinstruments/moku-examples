% Gigabit Streamer Example
%
% This example demonstrates how you can stream on a Moku in loopback,
% receiving and transmitting data using Moku Gigabit Streamer, using
% the Oscilloscope to generate and view the data in real-time.
%
% Connect a SFP cable in loopback from SFP1 to SFP2, to allow
%
% |   Slot  1   |   Slot  2   |   Slot  3   |
% |     gs1     |     osc     |     gs2     |
% |     rx      |   gen and   |     tx      |
% |  from SFP1  | view signal |   to SFP2   |
%
% (c) Liquid Instruments Pty. Ltd.

%% Connect to your Moku
% Connect to Moku via its IP address
% Change platform_id to 2 or 3 for Moku:Lab and Moku:Go, or 3 or 8 for Moku:Delta.
% force_connect will overtake an existing connection
m = MokuMultiInstrument('192.168.###.###', 3, force_connect=true);

try
    %% Configure the instruments
    % Set up Gigabit Streamer and Oscilloscope instruments
    gs1 = m.set_instrument(1, @MokuGigabitStreamer);   % Receiving Gigabit Streamer
    osc = m.set_instrument(2, @MokuOscilloscope);     % Oscilloscope
    gs2 = m.set_instrument(3, @MokuGigabitStreamer);   % Transmitting Gigabit Streamer

    % Set up connections between the instruments
    % The Oscilloscope is connected to view output from Gigabit Streamer slot 1 and
    % read back the generated signal
    connections = [struct('source', 'Slot1OutA', 'destination', 'Slot2InA');
                   struct('source', 'Slot2OutA', 'destination', 'Slot2InB');
                   struct('source', 'Slot2OutA', 'destination', 'Slot3InA');];
    m.set_connections(connections);

    %% Configure the Oscilloscope to generate and view the signal in real-time
    % Generate a 1 MHz sine wave, set trigger and timebase
    osc.generate_waveform(1,'Sine', 'amplitude', 1, 'frequency', 1e6, 'offset', 0, 'phase', 0);
    osc.set_trigger('type', 'Edge', 'source', 'ChannelA', 'level', 0);
    osc.set_timebase(-5e-6, 5e-6);

    %% Set up the receiving Gigabit Streamer instrument
    % Configure the acquisition settings
    gs1.enable_input(1, 'enable', true);
    gs1.enable_input(2, 'enable', false);
    gs1.set_acquisition('Normal', 156.25e6);
    gs1.set_interpolation('Linear');

    % Configure the local and remote network settings for gs1
    gs1.set_local_network('10.10.1.1', 5000);
    % Get the MAC address of Gigabit Streamer 2 (connected to SFP2)
    result = gs2.set_local_network('10.10.1.2', 5000);
    gs2_mac = result.mac_address;
    gs1.set_remote_network('10.10.1.2', 5000, gs2_mac);
    gs1.set_outgoing_packets('1500bytes');

    %% Set up the transmitting Gigabit Streamer instrument
    % Configure the acquisition settings
    gs2.enable_input(1, 'enable', true);
    gs2.enable_input(1, 'enable', true); % false
    gs2.set_acquisition('Normal', 156.25e6);
    gs2.set_interpolation('Linear');


    % Configure the local and remote network settings for gs2
    gs2.set_local_network('10.10.1.2', 5000);
    % Get the MAC address of Gigabit Streamer 1 (connected to SFP1)
    result = gs1.set_local_network('10.10.1.1', 5000);
    gs1_mac = result.mac_address;
    gs2.set_remote_network('10.10.1.1', 5000, gs1_mac);
    gs2.set_outgoing_packets('1500bytes');

    % Enable outputs and start sending data
    gs1.set_output(1, true, 0.0, 0.0);
    gs1.set_output(2, false, 0.0, 0.0);
    gs2.set_output(1, true, 0.0, 0.0);
    gs2.set_output(2, false, 0.0, 0.0);

    % Immediately start sending data for 10 seconds
    gs1.start_sending(10);
    gs2.start_sending(10);

    % Get initial data frame to set up plotting parameters
    data = osc.get_data();

    % Set up the plots
    figure
    lh = plot(data.time, data.ch1, data.time, data.ch2);
    xlabel(gca,'Time (sec)')
    ylabel(gca,'Amplitude (V)')

    %% Receive and plot new data frames
    while 1
        data = osc.get_data();
        set(lh(1),'XData',data.time,'YData',data.ch1);
        set(lh(2),'XData',data.time,'YData',data.ch2);

        axis tight
        pause(0.1)
    end

catch ME
    % End the current connection session with your Moku
    m.relinquish_ownership();
    rethrow(ME)
end
m.relinquish_ownership();
