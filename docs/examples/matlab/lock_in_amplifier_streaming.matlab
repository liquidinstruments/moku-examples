%% Plotting Lock-in Amplifier Example
%
%  This example demonstrates how you can configure the Lock-in Amplifier
%  instrument to demodulate an input signal from Input 1 with the reference
%  signal from the Local Oscillator to extract the X component and generate
%  a sine wave on the auxiliary output
%
%  (c) 2022 Liquid Instruments Pty. Ltd.
%

%% Connect to your Moku
% Connect to your Moku by its IP address and deploy the Lock-in Amplifier 
% instrument.
i = MokuLockInAmp('192.168.###.###');

try
    %% Configure the instrument
    
    % Configure the frontend
    % Channel 1 DC coupled, 1 MOhm impedance, and 400 mVpp range
    i.set_frontend(1, 'DC', '1MOhm','0dB');
    % Channel 2 DC coupled, 1 MOhm impedance, and 4 Vpp range
    i.set_frontend(2, 'DC', '1MOhm','0dB');
    
    % Configure the demodulation signal to Local oscillator with 1 MHz and
    % 0 degrees phase shift
    i.set_demodulation('Internal','frequency',1e6,'phase',0);
    
    % Set low pass filter to 1 kHz corner frequency with 6 dB/octave slope
    i.set_filter(1e3,'slope','Slope6dB');
    
    % Configure output signals
    % X component to Output 1 
    % Aux oscillator signal to Output 2 at 1 MHz 500 mVpp
    i.set_outputs('X','Aux');
    i.set_aux_output(1e6,0.5);
    
    %% Set up signal monitoring
    % Configure monitor points to Input 1 and main output
    i.set_monitor(1,'Input1');
    i.set_monitor(2,'MainOutput');
    
    % Configure the trigger conditions
    % Trigger on Probe A, rising edge, 0V
    i.set_trigger('type','Edge', 'source','ProbeA', 'level',0);
    
    % View +- 1 ms i.e. trigger in the centre
    i.set_timebase(-1e-3,1e-3);
    
    i.start_streaming('duration', 10);
    %% Set up plots
    % Get initial data to set up plots
    data = i.get_stream_data();
    
    % Set up the plots
    figure
    lh = plot(data.time, data.ch1, data.time, data.ch2);
    xlabel(gca,'Time (sec)')
    ylabel(gca,'Amplitude (V)')
    
    %% Receive and plot new data frames
    while 1
        data = i.get_stream_data();
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


