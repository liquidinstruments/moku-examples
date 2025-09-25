%% Plotting Oscilloscope Example
%
%  This example demonstrates how you can configure the Oscilloscope instrument
% to retrieve a single frame of dual-channel voltage data.
%
%  (c) Liquid Instruments Pty. Ltd.
%

%% Connect to your Moku
% Connect to your Moku by its IP address.
% force_connect will overtake an existing connection
m = MokuOscilloscope('192.168.###.###', force_connect=true);

try
    
    %% Configure the instrument
    
    % Configure the frontend
    % Channel 1 DC coupled, 10Vpp range
    m.set_frontend(1, '1MOhm', 'DC', '10Vpp');
    % Channel 2 DC coupled, 50Vpp range
    m.set_frontend(2, '1MOhm', 'DC', '10Vpp');
    
    % Configure the trigger conditions
    % Trigger on input Channel 1, rising edge, 0V
    m.set_trigger('type',"Edge", 'source',"Input1", 'level',0);
    
    % View +- 1 ms i.e. trigger in the centre
    m.set_timebase(-1e-3,1e-3);
    
    % Generate a sine wave on Output 1
    % 0.5Vpp, 10kHz, 0V offset
    m.generate_waveform(1, 'Sine', 'amplitude',0.5, 'frequency', 10e3);
    
    % Generate a square wave on Output 2
    % 1Vpp, 20kHz, 0V offset, 50% duty cycle
    m.generate_waveform(2, 'Square', 'amplitude',1, 'frequency',20e3, 'duty', 50);
    
    % Set the data source of Channel 1 to be Input 1
    m.set_source(1,'Input1');
    % Set the data source of Channel 2 to Input 2
    m.set_source(2,'Input2');
    
    %% Retrieve data
    % Get one frame of dual-channel voltage data
    data = m.get_data();

catch ME
    % End the current connection session with your Moku
    m.relinquish_ownership();
    rethrow(ME)
end

m.relinquish_ownership();

