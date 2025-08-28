%% Basic Datalogger Example
%
%  This example demonstrates how you can configure the Datalogger instrument.
%
%  (c) Liquid Instruments Pty. Ltd.
%

%% Connect to your Moku
% Connect to your Moku and deploy the oscilloscope instrument
% force_connect will overtake an existing connection
i = MokuDatalogger('192.168.###.###', force_connect=true);

try

    % Enable precision mode
    i.set_acquisition_mode('mode','Precision');
    
    % Generate a sine wave on Channel 1
    % 1Vpp, 10kHz, 0V offset
    i.generate_waveform(1, 'Sine', 'amplitude',1, 'frequency',10e3);
    % Generate a square wave on Channel 2
    % 1Vpp, 10kHz, 0V offset, 50% duty cycle
    i.generate_waveform(2, 'Square', 'amplitude',1, 'frequency', 1e3, 'duty', 50);
    
    % Start the data logging session of 10 second and store the file on the RAM
    % Set the sample rate to 1 kSa/s
    logging_request = i.start_logging('duration',10, 'sample_rate',1e3);
    log_file = logging_request.file_name;
    
    % Set up to display the logging process
    progress = i.logging_progress();
    
    while progress.complete < 1
        fprintf('%d seconds remaining \n',progress.time_remaining)
        pause(1);
        progress = i.logging_progress();
    end
    
    % Download the log file from the Moku to "Users" folder
    % Moku:Go should be downloaded from "persist", Moku:Lab from "media",
    % and Moku:Delta and Moku:Pro from "ssd"
    i.download_file('persist',log_file,['C:\Users\' log_file]);

catch ME
    % End the current connection session with your Moku
    i.relinquish_ownership();
    rethrow(ME)
end

% End the current connection session with your Moku
i.relinquish_ownership();