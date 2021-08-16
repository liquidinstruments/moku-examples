%% PPSU Example
%
%  This example will demonstrate how to configure the power supply
%  units of the Moku:Go.
%
%  (c) 2021 Liquid Instruments Pty. Ltd.
%

try
    % Connect to your Moku by its IP address.
    % An instrument must be deployed to establish the connection with the
    % Moku, in this example we will use the Oscilloscope.
    i = MokuOscilloscope('192.168.###.###');
    
    % Configure Power Supply Unit 1 to 2 V and 0.1 A
    i.set_power_supply(1,'enable',true,'voltage',2,'current',0.1);
    
    % Read the current status of Power Supply Unit 1 
    disp(i.read_power_supply(1));

catch ME
    % End the current connection session with your Moku
    i.relinquish_ownership();
    rethrow(ME)
end

if ~isempty(ME)
    % End the current connection session with your Moku
    i.relinquish_ownership();
end