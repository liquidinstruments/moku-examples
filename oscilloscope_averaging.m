%% Oscilloscope Averaging Example
%
%  This example demonstrates how to average oscilloscope waveforms
%  with the MATLAB API.
%
%  (c) 2025 Liquid Instruments Pty. Ltd.
%

%% Connect to your Moku
% Connect to your Moku by its IP address.
i = MokuOscilloscope('192.168.XX.XX', force_connect=true);

try
    %% Configure the instrument

    % Set frontend settings
    i.set_frontend(1, '50Ohm', 'DC', '4Vpp');

    % Set channel source
    i.set_source(1, 'Input1');

    % Set timebase and maximum number of data points
    i.set_timebase(-5e-6, 5e-6, 'max_length', 1024);

    % Set acquisition mode
    i.set_acquisition_mode('mode', 'Precision');

    % Set trigger settings
    i.set_trigger('type', 'Edge', 'source', 'Input1', 'level', 0, ...
                  'mode', 'Auto', 'edge', 'Rising');

    % Generate waveform from Output 1
    i.generate_waveform(1, 'Sine', 'amplitude', 1, 'frequency', 1e6);

    %% Set up averaging buffer
    % data_history stores the most recent frames in a circular buffer.
    % frame_averages sets the number of frames to average.
    % data_len is the number of points returned (adjusted with max_length).
    % current_count keeps track of frames collected.
    data_len = 1024;
    frame_averages = 50;
    data_history = zeros(frame_averages, data_len);
    current_count = 0;

    %% Initialize plot
    figure('Position', [100, 100, 1000, 600]);
    ax = axes;
    lh = plot(ax, NaN, NaN, 'b-', 'LineWidth', 1, 'DisplayName', 'Average');
    xlim(ax, [-5e-6, 5e-6]);
    ylim(ax, [-1.5, 1.5]);
    xlabel(ax, 'Time (s)');
    ylabel(ax, 'Voltage (V)');
    title(ax, 'Rolling Average');
    legend(ax, 'show');
    grid(ax, 'on');
    ax.GridAlpha = 0.3;

    %% Receive and plot averaged data frames
    while true
        % Collect oscilloscope data
        data = i.get_data();

        % Extract time and voltage data
        t = data.time;
        v = data.ch1;

        % Circular buffer indexing.
        % This stores the most recent frames for averaging.
        idx = mod(current_count, frame_averages) + 1;
        data_history(idx, :) = v;
        current_count = current_count + 1;

        % Average over the last 'frame_averages' samples
        num_samples = min(current_count, frame_averages);
        averaged_signal = mean(data_history(1:num_samples, :), 1);

        % Update the plot
        set(lh, 'XData', t, 'YData', averaged_signal);
        title(ax, sprintf('Rolling Average (Iteration %d)', current_count));
        drawnow;
        pause(0.1);

        fprintf('Iteration %d - Using %d samples for average\n', ...
                current_count, num_samples);
    end

catch ME
    % End the current connection session with your Moku 
    i.relinquish_ownership();
    rethrow(ME);
end

i.relinquish_ownership();