%% Basic Laser Lock Box Example
%
%  This example demonstrates how you can configure the Laser Lock Box 
%  Instrument.
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
    
    % Set low pass filter to 100 kHz corner frequency with 6 dB/octave slope
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
    
catch ME
    % End the current connection session with your Moku
    m.relinquish_ownership();
    rethrow(ME)
end

m.relinquish_ownership();



