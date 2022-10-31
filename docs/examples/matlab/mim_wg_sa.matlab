%% Multi-instrument WaveformGenerator-SpectrumAnalyzer Example
%
%  This example demonstrates how you can configure the multi instrument mode
% with Waveform Generator in slot1 and SpectrumAnalyzer in slot2
%
%  (c) 2021 Liquid Instruments Pty. Ltd.
%

%% Connect to your Moku
% Configure multi-instrument with platofrm_id 2
m = MokuMultiInstrument('192.168.###.###', 2);

try

    %% Configure the instruments
    % WaveformGenrator in slot1
    % SpectrumAnalyzer in slot2
    wg = m.set_instrument(1, @MokuWaveformGenerator);
    sa = m.set_instrument(2, @MokuSpectrumAnalyzer);

    % configure routing
    connections = [struct('source', 'Input1', 'destination', 'Slot1InA');
                struct('source', 'Slot1OutA', 'destination', 'Slot2InA')];

    m.set_connections(connections);

    % configure frontend
    m.set_frontend(1, "1MOhm", "DC", "0dB");

    %% Configure waveform generator
    % generate waveform
    wg.generate_waveform(1, 'Sine', 'frequency', 10e5);

    % set the span
    sa.set_span(0, 10e5);

    % set rbw mode
    sa.set_rbw('Auto');

    %% Retrieve data
    % Get one frame of spectrum data
    data = sa.get_data();

    % Set up the plots
    figure
    lh = plot(data.frequency, data.ch1);
    xlabel(gca, 'Frequency (Hz)')
    ylabel(gca, 'Amplitude (dBm)')

    %% Receive and plot new data frames
    while 1
        data = sa.get_data();

        set(lh(1), 'XData', data.frequency, 'YData', data.ch1);

        axis tight
        pause(0.1)
    end

catch ME
    % End the current connection session with your Moku
    m.relinquish_ownership();
    rethrow(ME)
end

m.relinquish_ownership();
