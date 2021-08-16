%% Plotting Frequency Response Analyzer Example 
%
% This example demonstrates how you can generate output sweeps using the
% Frequency Response Analyzer instrument, and view transfer function data 
% in real-time.
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

try
    %% Conect to the Moku
    % Connect to your Moku by its IP address.
    i = MokuFrequencyResponseAnalyzer('192.168.###.###');

    %% Configure the instrument
    % Set output sweep amplitudes and offsets
    i.set_output(1, 1,'offset',0); % Channel 1, 1Vpp, 0V offset
    i.set_output(2, 1,'offset',0); % Channel 2, 1Vpp, 0V offset

    % Configure the measurement mode to In/Out
    i.measurement_mode('input_only',false);

    % Set output sweep configuration
    i.set_sweep('start_frequency',f_start,'stop_frequency',f_stop, 'num_points',points, ...
        'averaging_time',averaging_time, 'averaging_cycles',averaging_cycles, ...
        'settling_time', settling_time, 'settling_cycles',settling_cycles);

    %% Set up plots
    % Get initial data to set up plots
    data = i.get_data();

    % Set up the plots
    figure;

    magnitude_graph = subplot(2,1,1);
    ms = semilogx(magnitude_graph, data.ch1.frequency,data.ch1.magnitude, data.ch2.frequency, data.ch2.magnitude);
    xlabel(magnitude_graph,'Frequency (Hz)');
    ylabel(magnitude_graph,'Magnitude (dB)');
    set(gca, 'XScale', 'log')

    phase_graph = subplot(2,1,2);
    ps = semilogx(phase_graph, data.ch1.frequency,data.ch1.phase, data.ch2.frequency, data.ch2.phase);
    xlabel(phase_graph,'Frequency (Hz)');
    ylabel(phase_graph,'Phase (cyc)');
    set(gca, 'XScale', 'log')

    %% Receive and plot new data frames
    while 1
        data = i.get_data();
        set(ms(1),'XData',data.ch1.frequency,'YData',data.ch1.magnitude);
        set(ms(2),'XData',data.ch2.frequency,'YData',data.ch2.magnitude);
        set(ps(1),'XData',data.ch1.frequency,'YData',data.ch1.phase);
        set(ps(2),'XData',data.ch2.frequency,'YData',data.ch2.phase);
        axis(magnitude_graph,'tight');
        axis(phase_graph,'tight');
        pause(0.1);
    end

catch ME
    i.relinquish_ownership();
    rethrow(ME)
end

if ~isempty(ME)
    i.relinquish_ownership();
end
