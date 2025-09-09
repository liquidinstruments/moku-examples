%% Basic Frequency Response Analyzer Example 
%
% This example demonstrates how you can generate output sweeps using the
% Frequency Response Analyzer instrument and retrieve a single sweep frame.
%
%  (c) Liquid Instruments Pty. Ltd.
%

%% Define sweep parameters here for readability
f_start = 20e6;  % Hz
f_stop= 100;  % Hz
points = 512;
averaging_time = 1e-6;  % sec
settling_time = 1e-6;  % sec
averaging_cycles = 1;
settling_cycles = 1;

%% Connect to Moku
% Connect to your Moku using its IP address.
% force_connect will overtake an existing connection
m = MokuFrequencyResponseAnalyzer('192.168.###.###', force_connect=true);

try

    %% Configure the instrument
    % Set output sweep amplitudes and offsets
    m.set_output(1, 1,'offset',0); % Channel 1, 1Vpp, 0V offset
    m.set_output(2, 1,'offset',0); % Channel 2, 1Vpp, 0V offset

    % Configure measurement mode to In/Out
    m.measurement_mode('mode','InOut');

    % Set sweep configuration
    m.set_sweep('start_frequency',f_start,'stop_frequency',f_stop, 'num_points',points, ...
        'averaging_time',averaging_time, 'averaging_cycles',averaging_cycles, ...
        'settling_time', settling_time, 'settling_cycles',settling_cycles);

    %% Get data from Moku
    % Get a single sweep frame from the Moku 
    data = m.get_data();

catch ME
    % End the current connection session with your Moku
    m.relinquish_ownership();
    rethrow(ME);
end

% End the current connection session with your Moku
m.relinquish_ownership();
 
