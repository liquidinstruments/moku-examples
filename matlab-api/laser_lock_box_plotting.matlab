%% Plotting Laser Lock Box Example
%
%  This example demonstrates how you can configure the Laser Lock Box 
%  Instrument and monitor the signals at Input 1 and Input 2.
%
%  (c) Liquid Instruments Pty. Ltd.
%

%% Connect to your Moku
% Connect to your Moku by its IP address and deploy the Laser Lock Box
% instrument.
% force_connect will overtake an existing connection
m = MokuLaserLockBox('192.168.###.###', force_connect=true);

try
    %% Configure the instrument
    
    % Configure the frontend
    % Channel 1 DC coupled, 1 MOhm impedance, and 400 mVpp range
    m.set_frontend(1, 'DC', '1MOhm', 'gain', '0dB');
    % Channel 2 DC coupled, 1 MOhm impedance, and 4 Vpp range
    m.set_frontend(2, 'DC', '1MOhm', 'gain', '-20dB');
    
    % Configure the scan oscillator to a 10 Hz 500 mVpp positive ramp
    % signal from Output 1
    m.set_scan_oscillator('enable',true,'shape','PositiveRamp', ...
        'frequency',10,'amplitude',0.5,'output','Output1');
    
    % Configure demodulation signal to Local Oscillator at 1 MHz and no
    % phase shift
    m.set_demodulation('Internal','frequency',1e6,'phase',0);
    
    % Configure a 4th order low pass filter with 100 kHz corner frequency
    m.set_filter('shape','Lowpass','low_corner',100e3,'order',4);
    
    % Set the fast PID controller to -10 dB proportional gain and
    % integrator crossover frequency at 3 kHz
    m.set_pid_by_frequency(1,-10,'int_crossover',3e3);
    % Set the slow PID controller to -10 dB proportional gain and
    % integrator crossover frequency at 50 Hz
    m.set_pid_by_frequency(2,-10,'int_crossover',50);
    
    % Enable the output channels
    m.set_output(1,true,true);
    m.set_output(2,true,true);
    
    %% Set up signal monitoring
    % Configure monitor points to Input 1 and Input 2
    m.set_monitor(1,'Input1');
    m.set_monitor(2,'Input2');
    
    % Configure the trigger conditions
    % Trigger on Probe A, rising edge, 0V
    m.set_trigger('type','Edge', 'source','ProbeA', 'level',0);
    
    % View +- 1 ms i.e. trigger in the centre
    m.set_timebase(-1e-3,1e-3);
    
    %% Set up plots
    % Get initial data to set up plots
    data = m.get_data();
    
    % Set up the plots
    figure
    lh = plot(data.time, data.ch1, data.time, data.ch2);
    xlabel(gca,'Time (sec)')
    ylabel(gca,'Amplitude (V)')
    
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



