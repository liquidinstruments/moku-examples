%% Basic Frequency Response Analyzer Example 
%
% This example demonstrates how you can generate output sweeps using the
% Frequency Response Analyzer instrument and retrieve a single sweep frame.
%
% (c) 2021 Liquid Instruments Pty. Ltd.
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
i = MokuFrequencyResponseAnalyzer('192.168.###.###');

try

    %% Configure the instrument
    % Set output sweep amplitudes and offsets
    i.set_output(1, 1,'offset',0); % Channel 1, 1Vpp, 0V offset
    i.set_output(2, 1,'offset',0); % Channel 2, 1Vpp, 0V offset

    % Configure measurement mode to In/Out
    i.measurement_mode('mode','InOut');

    % Set sweep configuration
    i.set_sweep('start_frequency',f_start,'stop_frequency',f_stop, 'num_points',points, ...
        'averaging_time',averaging_time, 'averaging_cycles',averaging_cycles, ...
        'settling_time', settling_time, 'settling_cycles',settling_cycles);

    %% Get data from Moku
    % Get a single sweep frame from the Moku 
    data = i.get_data();

catch ME
    % End the current connection session with your Moku
    i.relinquish_ownership();
    rethrow(ME);
end

% End the current connection session with your Moku
i.relinquish_ownership();
 
