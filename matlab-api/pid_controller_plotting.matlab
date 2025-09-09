%% Plotting PID Controller Example
%
%  This example demonstrates how you can configure the PID Controller instrument,
%  and view triggered time-voltage data frames in real-time.
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
    
    
    % Place 2 monitor points, one at input 1, one at output 1
    m.set_monitor(1,'Input1');
    m.set_monitor(2,'Output1');
    % Configure the timebase to -2 ms and 2 ms
    m.set_timebase(-0.002,0.002);
    
    % Configure the trigger
    m.set_trigger('type',"Edge", 'source',"ProbeA", 'level',0);
    
    %% Set up plots
    % Get initial data to set up plots
    data = m.get_data();
    
    % Set up the plots
    figure
    lh = plot(data.time, data.ch1, data.time, data.ch2);
    xlabel(gca,'Time (sec)')
    ylabel(gca,'Amplitude (V)')
    grid on
    grid(gca,'minor')
    %% Receive and plot new data frames
    while 1
        data = m.get_data();
        set(lh(1),'XData',data.time,'YData',data.ch1);
        set(lh(2),'XData',data.time,'YData',data.ch2);
        axis tight
        pause(0.1)
    end

catch ME
    % End the current connection session with your Moku
    m.relinquish_ownership();
    rethrow(ME)
end

m.relinquish_ownership();