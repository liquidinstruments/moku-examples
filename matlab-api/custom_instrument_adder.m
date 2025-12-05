%% Adder Custom Instrument Example
%
% This example demonstrates how you can configure Custom Instrument, using
% Multi-Instrument mode, to add and subtract two input signals together and
% output the result to the Oscilloscope.
%
%  (c) Liquid Instruments Pty. Ltd.
%
%% Before running
% The 'Adder' example is located at:
% (https://github.com/liquidinstruments/moku-examples/tree/main/mc/Basic/Adder)
%
% Unzip the bitstream (.tar file) once downloaded, and the unzipped folder
% contains 2, 3, or 4 .bar files depending on Moku hardware. The bitstream path
% should point to this unzipped folder.

%% Connect to your Moku
% Connect to Moku via its IP address. Change platform_id to 2 or 3 for Moku:Lab and Moku:Go, or 3 or 8 for Moku:Delta.
% force_connect will overtake an existing connection
m = MokuMultiInstrument('192.168.###.###', 4, force_connect=true);

try
    %% Configure the instruments
    % Set the instruments and upload Custom Instrument bitstreams from your device
    % to your Moku
    bitstream = 'path/to/project/adder/unzipped_bitstream';
    mc = m.set_instrument(1, @MokuCustomInstrument, bitstream);
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

    %% Plot the acquired data and set up plotting parameters
    % Get initial data to set up plots
    data = osc.get_data('wait_complete', true);

    % Set up the plots
    figure
    plot(data.time, data.ch1);
    hold on
    plot(data.time, data.ch2);
    xlabel(gca, 'Time (sec)')
    ylabel(gca, 'Amplitude (V)')
    legend('Add', 'Subtract')
    grid on;
    axis tight;

catch ME
    % End the current connection session with your Moku
    m.relinquish_ownership();
    rethrow(ME)
end

m.relinquish_ownership();
