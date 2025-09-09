%% Basic Spectrum Analyzer Example  
%
%  This example demonstrates how you can configure the Spectrum Analyzer
%  instrument to retrieve a single spectrum data frame over a set frequency 
%  span.
%
%  (c) Liquid Instruments Pty. Ltd.
%

%% Connect to the Moku
% Connect to your Moku by its IP address.
% force_connect will overtake an existing connection
m = MokuSpectrumAnalyzer('192.168.###.###', force_connect=true);

try
    
    %% Configure the instrument
    
    % Generate a sine wave on Channel 1
    % 1Vpp, 1MHz, 0V offset
    m.sa_output(1, 1, 1e6);
    % Generate a sine wave on Channel 2
    % 2Vpp, 50kHz, 0V offset
    m.sa_output(2, 2, 50e3);
    
    % Configure the measurement span to from 10Hz to 10MHz
    m.set_span(10,10e6);
    % Use Blackman Harris window
    m.set_window('BlackmanHarris');
    % Set resolution bandwidth to automatic
    m.set_rbw('Auto');
    
    %% Retrieve data
    % Get one frame of spectrum data
    data = m.get_data();

catch ME
    % End the current connection session with your Moku
    m.relinquish_ownership();
    rethrow(ME)
end

m.relinquish_ownership();

