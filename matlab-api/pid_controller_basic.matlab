%% Basic PID Controller Example
%
%  This example demonstrates how you can configure the PID Controller instrument.
%
%  (c) Liquid Instruments Pty. Ltd.
%

% Connect to your Moku and deploy the PID controller instrument
% force_connect will overtake an existing connection
m = MokuPIDController('192.168.###.###', force_connect=true);

try
    
    %% Configure the PID controller
    % Configure the control matrix
    m.set_control_matrix(1,1,0);
    m.set_control_matrix(2,0,1);
    % Enable all input and output channels
    m.enable_input(1,true);
    m.enable_input(2,true);
    m.enable_output(1,true,true);
    m.enable_output(2,true,true);
    
    % Configure controller 1 by gain
    m.set_by_gain(1,'prop_gain',10, 'diff_gain',-5,'diff_corner',5e3 );
    % Configure controller 2 by frequency
    m.set_by_frequency(2, 'prop_gain', -5, 'int_crossover',100, 'int_saturation',10);
    
  
catch ME
    % End the current connection session with your Moku
    m.relinquish_ownership();
    rethrow(ME)
end

m.relinquish_ownership();