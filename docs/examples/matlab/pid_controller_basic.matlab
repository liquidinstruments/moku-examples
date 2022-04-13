%% Basic PID Controller Example
%
%  This example demonstrates how you can configure the PID Controller instrument.
%
%  (c) 2021 Liquid Instruments Pty. Ltd.
%

% Connect to your Moku and deploy the PID controller instrument
i = MokuPIDController('192.168.###.###');

try
    
    %% Configure the PID controller
    % Configure the control matrix
    i.set_control_matrix(1,1,0);
    i.set_control_matrix(2,0,1);
    % Enable all input and output channels
    i.enable_input(1,true);
    i.enable_input(2,true);
    i.enable_output(1,true,true);
    i.enable_output(2,true,true);
    
    % Configure controller 1 by gain
    i.set_by_gain(1,'prop_gain',10, 'diff_gain',-5,'diff_corner',5e3 );
    % Configure controller 2 by frequency
    i.set_by_frequency(2, 'prop_gain', -5, 'int_crossover',100, 'int_saturation',10);
    
  
catch ME
    % End the current connection session with your Moku
    i.relinquish_ownership();
    rethrow(ME)
end

if ~isempty(ME)
    % End the current connection session with your Moku
    i.relinquish_ownership();
end