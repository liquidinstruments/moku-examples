%% Example: Data Acquisition with Deep Memory Mode in the Moku Oscilloscope
%
% This example demonstrates how to acquire data using the deep memory mode in the Moku Oscilloscope.
%
% The 'save_high_res_buffer' command stores high-resolution channel buffer data in Moku's internal storage.
%
% Ensure that deep memory mode is enabled using the 'set_acquisition_mode' command before exporting high-res data.
%
% Logged data files can be retrieved from the following storage locations:
%   - 'persist' for Moku:Go
%   - 'tmp' for Moku:Lab
%   - 'ssd' for Moku:Pro
%   - 'ssd' for Moku:Delta
% Update the 'download_file' command accordingly to match the correct storage location.
%
% The parameters in the 'set_frontend' command should be configured to align with the specific hardware (Moku:Go, Moku:Lab, Moku:Pro, or Moku:Delta).


NUM_FRAMES = 1;

% force_connect will overtake an existing connection
m = MokuOscilloscope('192.168.###.###', force_connect=true); % Connect to your moku device by its IP address

try
 
    m.set_trigger('type',"Edge", 'source',"Input1", 'level', 0);

    %% View +-5 msec, i.e. trigger in the centre
    
    m.set_timebase(-5e-3, 5e-3);
    
    m.set_acquisition_mode('mode',"DeepMemory");
    m.get_samplerate()

    % Set the data source of Channel 1 to be Input 1
    m.set_frontend(1,'50Ohm','AC','400mVpp')
    m.set_source(1, 'Input1');

    m.set_source(2, 'None');
    m.set_source(3, 'None');
    m.set_source(4, 'None');

    data_temp = [];
    for iter = 1:NUM_FRAMES
        m.get_data('wait_complete', true);
        response = m.save_high_res_buffer();
        file_name_temp = "./high_res_data-" + string(datetime('now', 'Format','d-MMM-y-HH_mm_ss'));
        m.download_file('ssd', response.file_name, file_name_temp +".li");
        system("mokucli convert --format=mat " + file_name_temp +".li")
        load(file_name_temp + ".mat");
        if(iter == 1)
            data_temp = moku.data(:,2);
        else
            data_temp = data_temp + moku.data(:,2);
        end
    end
    figure (1);
    plot(moku.data(:,1), data_temp./NUM_FRAMES); % This plots the average of all acquired high-res frames.
    xlabel('Time [s]');
    ylabel('Amplitude [V]');
    grid on;

catch ME
    % End the current connection session with your Moku
    m.relinquish_ownership();
    rethrow(ME);
end

m.relinquish_ownership();
