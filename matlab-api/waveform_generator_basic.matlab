%% Basic Waveform Generator Example 
%
%  This example demonstrates how you can configure the Waveform Generator
%  instrument.
%
%  (c) Liquid Instruments Pty. Ltd.
%

%% Connect to your Moku
% Connect to your Moku by its IP address.
% force_connect will overtake an existing connection
m = MokuWaveformGenerator('192.168.###.###', force_connect=true);

try

    %% Configure the instrument
    % Generate a sine wave on Channel 1
    % 0.5Vpp, 1MHz, 0V offset
    m.generate_waveform(1, 'Sine','amplitude', 0.5, 'frequency',1e6,'offset',0);
    % Generate a square wave on Channel 2
    % 1Vpp, 10kHz, 0V offset, 50% duty cycle
    m.generate_waveform(2, 'Square', 'amplitude',1,'frequency',10e3, 'duty',50);

    % Phase sync between the two channels
    m.sync_phase();

catch ME
    % End the current connection session with your Moku
    m.relinquish_ownership();
    rethrow(ME)
end

m.relinquish_ownership();