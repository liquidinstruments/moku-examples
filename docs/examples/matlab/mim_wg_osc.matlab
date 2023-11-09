%% Multi-instrument WaveformGenerator-Oscilloscope Example
%
%  This example demonstrates how you can configure the multi instrument mode
% with Waveform Generator in slot1 and Oscilloscope in slot2
%  and view triggered time-voltage data frames in real-time.
%
%  (c) 2021 Liquid Instruments Pty. Ltd.
%

%% Connect to your Moku
% Configure multi-instrument with platofrm_id 2
m = MokuMultiInstrument('192.168.###.###', 2);

try

    %% Configure the instruments
    % WaveformGenrator in slot1
    % Oscilloscope in slot2
    wg = m.set_instrument(1, @MokuWaveformGenerator);
    osc = m.set_instrument(2, @MokuOscilloscope);

    % configure routing
    connections = [struct('source', 'Input1', 'destination', 'Slot1InA');
                struct('source', 'Slot1OutA', 'destination', 'Slot2InA');
                struct('source', 'Slot1OutA', 'destination', 'Slot2InB');
                struct('source', 'Slot2OutA', 'destination', 'Output1')];

    m.set_connections(connections);

    % configure frontend
    m.set_frontend(1, "1MOhm", "DC", "0dB");

    %% Configure waveform generator
    % generate waveform
    wg.generate_waveform(1, "Sine");

    % set the timebase
    osc.set_timebase(-5e-3, 5e-3);

    %% Set up plots
    % Get initial data to set up plots
    data = osc.get_data();

    % Set up the plots
    figure
    lh = plot(data.time, data.ch1, data.time, data.ch2);
    xlabel(gca, 'Time (sec)')
    ylabel(gca, 'Amplitude (V)')

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
