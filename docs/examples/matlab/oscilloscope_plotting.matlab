%% Plotting Oscilloscope Example
%
%  This example demonstrates how you can configure the Oscilloscope instrument,
%  and view triggered time-voltage data frames in real-time.
%
%  (c) 2021 Liquid Instruments Pty. Ltd.
%

try
    %% Connect to your Moku
    % Connect to your Moku by its IP address.
    i = MokuOscilloscope('192.168.###.###');
    
    
    %% Configure the instrument
    
    % Configure the frontend
    % Channel 1 DC coupled, 10Vpp range
    i.set_frontend(1, '1MOhm', 'DC', '10Vpp');
    % Channel 2 DC coupled, 50Vpp range
    i.set_frontend(2, '1MOhm', 'DC', '10Vpp');
    
    % Configure the trigger conditions
    % Trigger on input Channel 1, rising edge, 0V
    i.set_trigger('type',"Edge", 'source',"Input1", 'level',0);
    
    % View +- 1 ms i.e. trigger in the centre
    i.set_timebase(-1e-3,1e-3);
    
    % Generate a sine wave on Output 1
    % 0.5Vpp, 10kHz, 0V offset
    i.generate_waveform(1, 'Sine', 'amplitude',0.5, 'frequency', 10e3);
    
    % Generate a square wave on Output 2
    % 1Vpp, 20kHz, 0V offset, 50% duty cycle
    i.generate_waveform(2, 'Square', 'amplitude',1, 'frequency',20e3, 'duty', 50);
    
    % Set the data source of Channel 1 to be Input 1
    i.set_source(1,'Input1');
    % Set the data source of Channel 2 to Input 2
    i.set_source(2,'Input2');
    
    %% Set up plots
    % Get initial data to set up plots
    data = i.get_data();
    
    % Set up the plots
    figure
    lh = plot(data.time, data.ch1, data.time, data.ch2);
    xlabel(gca,'Time (sec)')
    ylabel(gca,'Amplitude (V)')
    
    %% Receive and plot new data frames
    while 1
        data = i.get_data();
        set(lh(1),'XData',data.time,'YData',data.ch1);
        set(lh(2),'XData',data.time,'YData',data.ch2);
    
        axis tight
        pause(0.1)
    end


catch ME
    % End the current connection session with your Moku
    i.relinquish_ownership();
    rethrow(ME)
end

if ~isempty(ME)
    % End the current connection session with your Moku
    i.relinquish_ownership();
end
