%% Basic Lock-in Amplifier Example
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
    % Channel 1 DC coupled, 1 Mohm impedance, and 400 mVpp range
    i.set_frontend(1, 'DC', '1MOhm','0dB');
    % Channel 2 DC coupled, 1 Mohm impedance, and 4 Vpp range
    i.set_frontend(2, 'DC', '1MOhm','-20dB');
    
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
    

catch ME
    % End the current connection session with your Moku
    i.relinquish_ownership();
    rethrow(ME)
end

i.relinquish_ownership();


