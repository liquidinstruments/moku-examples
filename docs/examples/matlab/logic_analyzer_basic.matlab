%% Basic Logic Analyzer Example 
%
%  This example demonstrates how you can configure the Logic
%  Analyzer instrument to retrieve a single frame of data for 
%  all 16 channels
%
%  (c) 2021 Liquid Instruments Pty. Ltd.
%
    
%% Connect to your Moku
% Connect to your Moku by its IP address.
i = MokuLogicAnalyzer('192.168.###.###');

try

    
    %% Configure the instrument
    
    % Configure the digital pins
    % - Pin 1 as an output pin
    % - Pin 2 as high
    % - Pin 3 as low
    % - Pin 4 as input
    % - Pin 5 as input
    % - Pin 6 as input
    i.set_pins("Pin1", 'O');
    i.set_pins("Pin2", 'H');
    i.set_pins("Pin3", 'L');
    i.set_pins("Pin4", 'I');
    i.set_pins("Pin5", 'I');
    i.set_pins("Pin6", 'I');
    
    % Configure Pin 1 output pattern to [1 1 0 0]
    i.generate_pattern('Pin1', [1 1 0 0]);
    
    % Start the IO pin outputs for all pins
    i.start_all();
    
    % Set trigger channel as Pin 5
    i.set_trigger("Pin5");
    
    % View +- 100 nanosecond i.e. trigger in the centre
    i.set_timebase(-100e-9,100e-9,false);
    
    %% Retrieve data
    % Get one frame of data for all 16 pins
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

