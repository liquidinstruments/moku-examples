%% Basic Datalogger Example
%
%  This example demonstrates how you can configure the Datalogger instrument.
%
%  (c) 2021 Liquid Instruments Pty. Ltd.
%

try
    %% Connect to your Moku
    % Connect to your Moku and deploy the oscilloscope instrument
    i = MokuDatalogger('192.168.###.###');
    
    % Enable precision mode
    i.set_acquisition_mode('mode','Precision');
    
    % Set the sample rate to 1 kSa/s
    i.set_samplerate(1e3);
    
    % Generate a sine wave on Channel 1
    % 1Vpp, 10kHz, 0V offset
    i.generate_waveform(1, 'Sine', 'amplitude',1, 'frequency',10e3);
    % Generate a square wave on Channel 2
    % 1Vpp, 10kHz, 0V offset, 50% duty cycle
    i.generate_waveform(2, 'Square', 'amplitude',1, 'frequency', 1e3, 'duty', 50);
    
    % Start the data logging session of 10 second and store the file on the RAM
    log_file = i.start_logging('duration',10);
    
    % Set up to display the logging process
    progress = i.logging_progress();
    
    while str2double(progress.time_to_end) > 2
        fprintf('%s seconds remaining \n',progress.time_to_end)
        pause(1);
        progress = i.logging_progress();
    end
    
    % Download the log file from the Moku to "Users" folder
    % Moku:Go should be downloaded from "persist" and Moku:Pro from "ssd"
    i.download_file('persist',log_file,['C:\Users\' log_file]);

catch ME
    % End the current connection session with your Moku
    i.relinquish_ownership();
    rethrow(ME)
end

if ~isempty(ME)
    % End the current connection session with your Moku
    i.relinquish_ownership();
end