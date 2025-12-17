% Gigabit Streamer+ Example
%
% This example demonstrates how you can transmit and receive data
% using Moku Gigabit Streamer+ in single instrument mode.
%
% (c) Liquid Instruments Pty. Ltd.

%% Connect to your Moku
% Connect to Moku via its IP address
% force_connect will overtake an existing connection
m = MokuGigabitStreamerPlus('192.168.XXX.XXX', force_connect=true);

try
    %% Set up the Gigabit Streamer+ inputs (channel, enable)
    m.enable_input(1, 'enable', true);
    m.enable_input(2, 'enable', true);
    m.enable_input(3, 'enable', true);
    m.enable_input(4, 'enable', true);

    % Configure the acquisition settings
    m.set_acquisition('Normal', 1.25e9);
    m.set_interpolation('Linear');

    % Set analog frontend settings (channel, impedance, coupling, gain)
    m.set_frontend(1, "50Ohm", "DC", 'gain', "0dB");
    m.set_frontend(2, "50Ohm", "DC", 'gain', "0dB");
    m.set_frontend(3, "50Ohm", "DC", 'gain', "0dB");
    m.set_frontend(4, "50Ohm", "DC", 'gain', "0dB");

    % Configure the local and remote network settings
    m.set_local_network('10.10.1.1', 5000);
    % Find host or configure the host IP address, UDP port,
    % and MAC address to set the remote network
    m.set_remote_network('168.192.XXX.XXX', 5000, 'A1:B2:C3:XX:XX:XX');
    m.set_outgoing_packets('1500bytes');

    % Enable outputs (channel, enable, gain, offset)
    m.set_output(1, true, 0.0, 0.0);
    m.set_output(2, true, 0.0, 0.0);
    m.set_output(3, true, 0.0, 0.0);
    m.set_output(4, true, 0.0, 0.0);

    % Print a summary of the instrument configuration
    m.summary

    % Immediately start sending data for 10 seconds
    m.start_sending(10);

catch ME
    % End the current connection session with your Moku
    m.relinquish_ownership();
    rethrow(ME)
end
m.relinquish_ownership();
