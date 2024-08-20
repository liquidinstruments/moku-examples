%% Basic Waveform Generator Example 
%
%  This example demonstrates how you can configure the Waveform Generator
%  instrument.
%
%  (c) 2021 Liquid Instruments Pty. Ltd.
%

%% Connect to your Moku
% Connect to your Moku by its IP address.
i = MokuWaveformGenerator('192.168.###.###');

try

    %% Configure the instrument
    % Generate a sine wave on Channel 1
    % 0.5Vpp, 1MHz, 0V offset
    i.generate_waveform(1, 'Sine','amplitude', 0.5, 'frequency',1e6,'offset',0);
    % Generate a square wave on Channel 2
    % 1Vpp, 10kHz, 0V offset, 50% duty cycle
    i.generate_waveform(2, 'Square', 'amplitude',1,'frequency',10e3, 'duty',50);

    % Phase sync between the two channels
    i.sync_phase();

catch ME
    % End the current connection session with your Moku
    i.relinquish_ownership();
    rethrow(ME)
end

i.relinquish_ownership();