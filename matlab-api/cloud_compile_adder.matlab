%% Adder Cloud Compile Example
%
%  This example demonstrates how you can configure Cloud Compile, using
% Multi-Instrument mode, to add and subtract two input signals together and
% output the result to the Oscilloscope.
%
%  (c) Liquid Instruments Pty. Ltd.
%

%% Connect to your Moku
% Configure multi-instrument with platform_id 2
% force_connect will overtake an existing connection
m = MokuMultiInstrument('192.168.###.###', 2, force_connect=true);

try
    %% Configure the instruments
    % Set the instruments and upload Cloud Compile bitstreams from your device
    % to your Moku
    bitstream = 'path/to/project/adder/bitstreams.tar';
    mcc = m.set_instrument(1, @MokuCloudCompile, bitstream);
    osc = m.set_instrument(2, @MokuOscilloscope);

    % configure routing
    connections = [struct('source','Slot1OutA', 'destination','Slot2InA');
                   struct('source','Slot1OutB', 'destination','Slot2InB');
                   struct('source','Slot2OutA', 'destination','Slot1InA');
                   struct('source','Slot2OutB', 'destination','Slot1InB')];

    m.set_connections(connections);

    %% Configure the Oscilloscope to generate a ramp wave and square wave with
    % equal frequencies, then sync the phases
    osc.generate_waveform(1, 'Square', 'amplitude',1, 'frequency',1e3, 'duty',50);
    osc.generate_waveform(2, 'Ramp', 'amplitude',1, 'frequency',1e3, 'symmetry',50);
    osc.sync_output_phase();

    % Set the time span to cover four cycles of the waveforms
    osc.set_timebase(-2e-3, 2e-3);

    %% Set up plots
    % Get initial data to set up plots
    data = osc.get_data();

    % Set up the plots
    figure
    lh = plot(data.time, data.ch1, data.time, data.ch2);
    xlabel(gca, 'Time (sec)')
    ylabel(gca, 'Amplitude (V)')
    legend('Add', 'Subtract')

    %% Receive and plot new data frames
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
