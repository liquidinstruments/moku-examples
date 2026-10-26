%% Multi-instrument WaveformGenerator-RealTimeMath-Oscilloscope Example
%
%  This example demonstrates the Real Time Math (RTM) instrument in
%  Multi-Instrument Mode. A Waveform Generator in slot1 drives RTM in slot2,
%  and an Oscilloscope in slot3 shows the two RTM outputs:
%    - Output A: the signal amplified by 6 dB, with the output limited to +/-0.5 V
%    - Output B: the signal with 0.25 V added to it
%
%  (c) Liquid Instruments Pty. Ltd.
%

%% Connect to your Moku
% RTM only runs in Multi-Instrument Mode. This example uses three slots, so
% use num_slots 3 for Moku:Go, Moku:Lab and Moku:Delta, or 4 for Moku:Pro.
% force_connect will overtake an existing connection
m = MokuMultiInstrument('192.168.###.###', 3, force_connect=true);

try

    %% Configure the instruments
    % WaveformGenerator in slot1
    % RealTimeMath in slot2
    % Oscilloscope in slot3
    wg = m.set_instrument(1, @MokuWaveformGenerator);
    rtm = m.set_instrument(2, @MokuRealTimeMath);
    osc = m.set_instrument(3, @MokuOscilloscope);

    % configure routing: WaveformGenerator -> RTM -> Oscilloscope
    connections = [struct('source', 'Slot1OutA', 'destination', 'Slot2InA');
                struct('source', 'Slot2OutA', 'destination', 'Slot3InA');
                struct('source', 'Slot2OutB', 'destination', 'Slot3InB')];

    m.set_connections(connections);

    %% Configure waveform generator
    % a 1 Vpp, 10 kHz sine wave
    wg.generate_waveform(1, "Sine", 'amplitude', 1, 'frequency', 10e3);

    %% Configure real time math
    % Each RTM output channel applies a function to the input signals.
    % Output A: gain of 6 dB (about x2), limited to +/-0.5 V so the peaks are clipped
    rtm.set_node(1, 1, 1, "Gain", 'gain', 6);
    rtm.set_output_limit(1, true, 'low_limit', -0.5, 'high_limit', 0.5);

    % Output B: add a 0.25 V offset to the signal
    rtm.set_node(2, 1, 1, "Offset", 'offset', 0.25);

    % print the RTM configuration
    disp(rtm.summary());

    %% View both outputs on the oscilloscope
    osc.set_timebase(-0.5e-3, 0.5e-3);
    data = osc.get_data();

    figure
    plot(data.time, data.ch1, data.time, data.ch2);
    grid on
    xlabel(gca, 'Time (sec)')
    ylabel(gca, 'Amplitude (V)')
    legend('RTM output A (gain, limited)', 'RTM output B (offset)')

catch ME
    % End the current connection session with your Moku
    m.relinquish_ownership();
    rethrow(ME)
end

m.relinquish_ownership();
