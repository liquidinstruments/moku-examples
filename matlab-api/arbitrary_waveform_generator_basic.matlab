%% Arbitrary Waveform Generator Example 
%
%  This example demonstrates how you can configure the Arbitrary Waveform 
%  Generator instrument to generate two signals. 
%
%  (c) Liquid Instruments Pty. Ltd.
%
% 
%% Prepare the waveforms
% Prepare the square waveform to be generated
t = linspace(0,1,100);
square_wave = sign(sin(2*pi*t));

% Prepare a more interesting waveform to be generated (note that the points
% must be normalized to range [-1,1])
not_square_wave = zeros(1,length(t));
for h=1:2:15
    not_square_wave = not_square_wave + (4/pi*h)*cos(2*pi*h*t);
end
not_square_wave = not_square_wave / max(not_square_wave);

%% Connect to your Moku
% Connect to your Moku by its IP address.
   
% force_connect will overtake an existing connection
m = MokuArbitraryWaveformGenerator('192.168.###.###', force_connect=true);

try
 
    % Configure the output waveform in each channel
    % Channel 1: sampling rate of 125 MSa/s, square wave, 1MHz, 2Vpp.
    m.generate_waveform(1, "625Ms", square_wave, 1e6, 2);
    % Channel 2: automatic sampling rate, use the "not_square_wave" LUT, 10
    % kHz, 1Vpp.
    m.generate_waveform(2, "Auto", not_square_wave, 10e3, 1,...
        'interpolation', true);
    
    %% Set channel 1 to pulse mode 
    % 2 dead cycles at 0Vpp
    m.pulse_modulate(1,'dead_cycles',2,'dead_voltage',0);

    %% Set Channel 2 to burst mode
    % Burst mode triggering from Input 1 at 0.1 V
    % 3 cycles of the waveform will be generated every time it is triggered
    m.burst_modulate(2, "Input1", "NCycle",'burst_cycles',3,'trigger_level',0.1);

catch ME
    % End the current connection session with your Moku
    m.relinquish_ownership();
    rethrow(ME);
end

m.relinquish_ownership();

