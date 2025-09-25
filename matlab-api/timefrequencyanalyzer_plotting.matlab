%% Plotting Time and Frequency Analyzer Example
%
%  This example demonstrates how you can configure the Time and Frequency Analyzer
% instrument, and view histogram data frames in real-time.
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

    % Set the histogram to 8ns span centred around 2us
    m.set_histogram(1.996e-6, 2.004e-6);

    % Set the interval analyzers
    % Set Interval A to start at Event A and stop at Event A
    % Set Interval B to start at Event B and stop at Event B
    m.set_interval_analyzer(1, 1, 1);
    m.set_interval_analyzer(2, 2, 2);

    %% Retrieve data
    % Get initial data to set up plots
    data = m.get_data();

    % Set up the plots
    figure
    xlabel(gca,'Time (sec)')
    ylabel(gca,'Count')

    x = linspace(data.interval1.histogram.t0, data.interval1.histogram.t0 + 1023 * data.interval1.histogram.dt, 1024);
    lh = bar(x, [data.interval1.histogram.data data.interval2.histogram.data]);

    %% Receive and plot new data frames
    while 1
        data = m.get_data();
        set(lh(1),'YData',data.interval1.histogram.data);
        set(lh(2),'YData',data.interval2.histogram.data);

        axis tight
        pause(0.1)
    end

catch ME
    % End the current connection session with your Moku
    m.relinquish_ownership();
    rethrow(ME)
end

m.relinquish_ownership();
