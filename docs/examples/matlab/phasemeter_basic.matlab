%% Basic Phasemeter Example
%
%  This example demonstrates how you can configure the Phasemeter 
%  instrument to measure 4 independent signals
%
%  (c) 2021 Liquid Instruments Pty. Ltd.
%
try
    %% Connect to your Moku
    % Connect to your Moku and deploy the oscilloscope instrument
    i = MokuPhasemeter('192.168.###.###');
    
    % Configure all Output channels to generate sine waves at 1 Vpp, 2 MHz
    i.generate_output(1,1,2e6);
    i.generate_output(2,1,2e6);
    i.generate_output(3,1,2e6);
    i.generate_output(4,1,2e6);
    
    % Set the acquisition speed to 480 Hz
    i.set_acquisition_speed('480Hz');
    
    % Set all input channels to 2 MHz, bandwidth 40 Hz
    i.set_pm_loop(1,'auto_acquire',false,'frequency',2e6,'bandwidth','40Hz');
    i.set_pm_loop(1,'auto_acquire',false,'frequency',2e6,'bandwidth','40Hz');
    i.set_pm_loop(1,'auto_acquire',false,'frequency',2e6,'bandwidth','40Hz');
    i.set_pm_loop(1,'auto_acquire',false,'frequency',2e6,'bandwidth','40Hz');
    
    % Set frontend of Channel 1 and Channel 2 to 50 Ohm, DC coupled, 4 Vpp
    % range
    i.set_frontend(1,'50Ohm','DC','4Vpp');
    i.set_frontend(2,'50Ohm','DC','4Vpp');
    
    % Get all the data available from the Moku
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