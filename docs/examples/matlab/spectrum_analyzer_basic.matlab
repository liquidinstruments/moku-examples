%% Basic Spectrum Analzyer Example  
%
%  This example demonstrates how you can configure the Spectrum Analyzer
%  instrument to retrieve a single spectrum data frame over a set frequency 
%  span.
%
%  (c) 2021 Liquid Instruments Pty. Ltd.
%

try
    %% Conect to the Moku
    % Connect to your Moku by its IP address.
    i = MokuSpectrumAnalyzer('192.168.###.###');
    
    %% Configure the instrument
    
    % Generate a sine wave on Channel 1
    % 1Vpp, 1MHz, 0V offset
    i.sa_output(1, 1, 1e6);
    % Generate a sine wave on Channel 2
    % 2Vpp, 50kHz, 0V offset
    i.sa_output(2, 2, 50e3);
    
    % Configure the measurement span to from 10Hz to 10MHz
    i.set_span(10,10e6);
    % Use Blackman Harris window
    i.set_window('BlackmanHarris');
    % Set resolution bandwidth to automatic
    i.set_rbw('Auto');
    
    %% Retrieve data
    % Get one frame of spectrum data
    data = i.get_data();

catch ME
    % End the current connection session with your Moku
    i.relinquish_ownership();
    rethrow(ME)
end

if ~isempty(ME)
    % End the current connection session with your Moku
    i.relinquish_ownership();
end

