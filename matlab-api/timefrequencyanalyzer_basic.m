%% Plotting Time and Frequency Analyzer Example
%
%  This example demonstrates how you can configure the Time and Frequency Analyzer
% instrument, and view the statistics of the intervals.
%
%  (c) Liquid Instruments Pty. Ltd.
%

%% Connect to your Moku
% Connect to your Moku by its IP address.
% force_connect will overtake an existing connection
m = MokuTimeFrequencyAnalyzer('192.168.###.###', force_connect=true);

try

    %% Configure the instrument

    % Set the event detectors
    % Set Event A to detect rising edge events on Input 1 at 0V
    % Set Event B to detect rising edge events on Input 2 at 0V
    m.set_event_detector(1, 'Input1', 'threshold',0, 'edge','Rising');
    m.set_event_detector(2, 'Input2', 'threshold',0, 'edge','Rising');

    % Set the interpolation to Linear
    m.set_interpolation('Linear');

    % Set acquisition mode to a 100ms Window
    m.set_acquisition_mode('Windowed', 'window_length',100e-3);

    % Use the first start event and close the interval at the end of the gate period (arbitrary as using windowed acquisition)
    m.set_interval_policy('Use first', 'Close');

    % Set the interval analyzers
    % Set Interval A to start at Event A and stop at Event A
    % Set Interval B to start at Event B and stop at Event B
    m.set_interval_analyzer(1, 1, 1);
    m.set_interval_analyzer(2, 2, 2);

    %% Retrieve data
    % Get data and explore statistics
    data = m.get_data();
    disp('Interval 1')
    disp(data.interval1.statistics)
    disp('Interval 2')
    disp(data.interval2.statistics)

catch ME
    % End the current connection session with your Moku
    m.relinquish_ownership();
    rethrow(ME)
end

m.relinquish_ownership();
