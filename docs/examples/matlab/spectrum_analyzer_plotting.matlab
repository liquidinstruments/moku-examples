%% Plotting Spectrum Analzyer Example 
%
%  This example demonstrates how you can configure the Spectrum Analyzer
%  instrument, and view measured spectrum in real-time.
%
%
%  (c) 2021 Liquid Instruments Pty. Ltd.
%
%% Conect to the Moku
% Connect to your Moku by its IP address.
i = MokuSpectrumAnalyzer('192.168.###.###');

%% Configure the instrument

% Generate a sine wave on Channel 1
% 1Vpp, 1MHz, 0V offset
i.sa_output(1, 1, 1e6);
% Generate a sine wave on Channel 2
% 2Vpp, 50kHz, 0V offset
i.sa_output(2, 2, 50e3);

% Configure the measurement span to from 10Hz to 10MHz
i.set_span(10,10e6);
% Use Blackman Harris window
i.set_window(1,'BlackmanHarris');
i.set_window(2,'BlackmanHarris');
% Set resolution bandwidth to automatic
i.set_rbw('Auto');

%% Set up plots
% Get initial data to set up plots
data = i.get_data();

% Set up the plots
figure
lh = plot(data.frequency, data.ch1, data.frequency, data.ch2);
xlabel(gca,'Frequency (Hz)')
ylabel(gca,'Amplitude (dBm)')

%% Receive and plot new data frames
while 1
    data = i.get_data();

    set(lh(1),'XData',data.frequency,'YData',data.ch1);
    set(lh(2),'XData',data.frequency,'YData',data.ch2);

    axis tight
    pause(0.1)
end


