%% Plotting Digital Filter Box Example
%
% This example demonstrates how you can configure the Digital
% Filter Box instrument to filter and display two signals. Filter
% 1 takes its input from Input1 and applies a lowpass Butterworth
% filter. Filter 2 takes its input from Input1 + Input2 and applies
% a highpass Elliptic filter. The output of both filters are then
% displayed on the plot.
%
%  (c) 2024 Liquid Instruments Pty. Ltd.
%

%% Connect to the Moku
% Connect to your Moku by its IP address.
i = MokuDigitalFilterBox('192.168.###.###');

try
    % Set Channel 1 and 2 to DC coupled, 1 MOhm impedance, and
    % default input range (400 mVpp range on Moku:Pro, 1 Vpp
    % range on Moku:Lab, 10 Vpp range on Moku:Go)
    i.set_frontend(1, 'DC', '1MOhm', '0dB');
    i.set_frontend(2, 'DC', '1MOhm', '0dB');

    % Channel1 signal: Input 1
    % Channel2 signal: Input 1 + Input 2
    i.set_control_matrix(1, 1, 0);
    i.set_control_matrix(2, 1, 1);

    % Configure Digital Filter on Channel 1 and Channel 2
    % Channel1 is a 8th-order Butterworth lowpass filter
    % Channel2 is a 8th-order Elliptic highpass filter
    % 3.906 MHz is for Moku:Go
    % Please change sampling rate for other Moku devices.
    i.set_filter(1, "3.906MHz", 'shape',"Lowpass", 'type',"Butterworth",...
        'low_corner',1e3, 'order',8);
    i.set_filter(1, "3.906MHz", 'shape',"Highpass", 'type',"Elliptic",...
        'high_corner',100e3, 'order',8);

    % Monitor ProbeA: Filter Channel1 output
    % Monitor ProbeB: Filter Channel2 output
    i.set_monitor(1, "Output1");
    i.set_monitor(2, "Output2");

    % Enable Digital Filter Box output ports
    i.enable_output(1, 'signal',true, 'output',true);
    i.enable_output(2, 'signal',true, 'output',true);

    % View +- 0.5 ms i.e. trigger in the centre
    i.set_timebase(-0.5e-3, 0.5e-3);
    
    %% Retrieve data
    % Get one frame of data
    data = i.get_data();
    
    % Set up the plots
    figure
    lh = plot(data.time, data.ch1, data.time, data.ch2);
    xlabel(gca,'Time (s)')
    ylabel(gca,'Amplitude (V)')
    legend('Lowpass Filter Output', 'Highpass Filter Output')
    %% Receive and plot new data frames
    while 1
        data = i.get_data();
    
        set(lh(1),'XData',data.time,'YData',data.ch1);
        set(lh(2),'XData',data.time,'YData',data.ch2);
    
        axis tight
        pause(0.1)
    end

catch ME
    % End the current connection session with your Moku
    i.relinquish_ownership();
    rethrow(ME)
end

i.relinquish_ownership();


