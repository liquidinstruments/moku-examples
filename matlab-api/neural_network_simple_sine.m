% 
% moku example: Neural Network
%
% This script demonstrates how to use the Neural Network instrument to generate
% a ramp wave and process it through the uploaded neural network, finally 
% viewing the output in the oscilloscope. This uses the network generated 
% in the Simple Sine wave example.
%
%  (c) Liquid Instruments Pty. Ltd.

% force_connect will overtake an existing connection
m = MokuMultiInstrument('192.168.###.###', 4, force_connect=true, timeout=60);

wg = m.set_instrument(1, @MokuWaveformGenerator);
nn = m.set_instrument(2, @MokuNeuralNetwork);
osc = m.set_instrument(3, @MokuOscilloscope);

try
% configure routing
connections = [struct(source="Slot1OutA", destination="Slot2InA");
               struct(source="Slot1OutA", destination="Slot2InB");
               struct(source="Slot1OutB", destination="Slot2InC");
               struct(source="Slot1OutB", destination="Slot2InD");
               struct(source="Slot2OutA", destination="Slot3InA");
               struct(source="Slot2OutB", destination="Slot3InB");
               struct(source="Slot2OutC", destination="Slot3InC");
               struct(source="Slot2OutD", destination="Slot3InD")];

m.set_connections(connections)

% Generate ramp wave in Waveform Generator
wg.generate_waveform(1, 'Ramp', 'amplitude', 0.5, 'frequency', 5e3, 'symmetry', 100)
wg.summary()

% Load network into Neural Network and set inputs and outputs
nn.set_input(1, -1, 1);
nn.set_input_sample_rate(305000);
nn.upload_network("/path/to/simple_sine_model.linn");
nn.set_output(1, true, 'channel', 1, 'low_level', 0, 'high_level', 1);
nn.summary()


% Set up Oscilloscope for plotting
osc.set_timebase(-5e-3, 5e-3);

% Get initial data to set up plots
data = osc.get_data();
osc.summary()

% Set up the plots
figure
lh = plot(data.time, data.ch1, data.time, data.ch2);
xlabel(gca, 'Time (sec)')
ylabel(gca, 'Amplitude (V)')

% Receive and plot new data frames
while 1
    data = osc.get_data();
    set(lh(1), 'XData', data.time, 'YData', data.ch1);
    set(lh(2), 'XData', data.time, 'YData', data.ch2);

    axis tight
    pause(0.1)
end


catch ME
    % End the current connection session with your Moku
    m.relinquish_ownership();
    rethrow(ME)
end

m.relinquish_ownership();
