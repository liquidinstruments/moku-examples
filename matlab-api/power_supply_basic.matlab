%% PPSU Example
%
%  This example will demonstrate how to configure the power supply
%  units of the Moku:Go.
%
%  (c) Liquid Instruments Pty. Ltd.
%

% Connect to your Moku by its IP address.
% An instrument must be deployed to establish the connection with the
% Moku, in this example we will use the Oscilloscope.
% force_connect will overtake an existing connection
m = MokuOscilloscope('192.168.###.###', force_connect=true);
try

    % Configure Power Supply Unit 1 to 2 V and 0.1 A
    m.set_power_supply(1,'enable',true,'voltage',2,'current',0.1);
    
    % Read the current status of Power Supply Unit 1 
    disp(m.get_power_supply(1));

catch ME
    % End the current connection session with your Moku
    m.relinquish_ownership();
    rethrow(ME)
end

m.relinquish_ownership();