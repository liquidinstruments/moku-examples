%% Datalogger streaming Example
%
%  This example demonstrates how you can configure the Datalogger instrument,
%  and view triggered time-voltage data stream in real-time.
%
%  (c) Liquid Instruments Pty. Ltd.
%

%% Connect to your Moku
% Connect to your Moku by its IP address.
% force_connect will overtake an existing connection
i = MokuDatalogger('192.168.###.###', force_connect=true);

try

    % Generate a sine wave on Output 1
    % 0.5Vpp, 10kHz, 0V offset
    i.generate_waveform(1, 'Sine', 'amplitude', 0.5, 'frequency', 10e3);

    % Generate a square wave on Output 2
    % 1Vpp, 20kHz, 0V offset, 50% duty cycle
    i.generate_waveform(2, 'Square', 'amplitude', 1, 'frequency', 20e3, 'duty', 50);
    
    % Start the data streaming session of 30 second and
    % set the sample rate to 1 kSa/s
    i.start_streaming('duration', 30, 'sample_rate',1e3);

  
    %% Set up plots
    % Get initial data to set up plots
    data = i.get_stream_data();
    
    % Set up the plots
    figure
    lh = plot(data.time, data.ch1, data.time, data.ch2);
    xlabel(gca,'Time (sec)')
    ylabel(gca,'Amplitude (V)')
    
    %% Receive and plot new data frames
    while 1
        data = i.get_stream_data();
        set(lh(1),'XData',data.time,'YData',data.ch1);
        set(lh(2),'XData',data.time,'YData',data.ch2);
    
        axis tight
        pause(0.1)
    end

catch ME
    if strcmp(ME.message, 'End of stream')
        disp('Streaming session complete!');
    else
        i.stop_streaming();
        % End the current connection session with your Moku
        i.relinquish_ownership();
        rethrow(ME)
    end
end

i.relinquish_ownership();
