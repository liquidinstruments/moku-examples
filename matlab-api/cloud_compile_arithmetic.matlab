%% Arithmetic Cloud Compile Example
%
%  This example demonstrates how you can configure Cloud Compile to choose to
% add, subtract or multiply two input signals using the control registers
% and output the result to the Oscilloscope.
%
%  (c) Liquid Instruments Pty. Ltd.
%
%% Before running
% The 'Arithmetic Unit' example is located at:
% (https://github.com/liquidinstruments/moku-examples/tree/main/mcc/Moderate/ArithmeticUnit#arithmetic-unit-example)
%
% Unzip the bitstream (.tar file) once downloaded, and the unzipped folder
% contains 2 or 4 .bar files depending on Moku hardware. The bitstream path
% should point to this unzipped folder.

%% Connect to your Moku
% Connect to Moku via its IP address. Change platform_id to 2 for Moku:Lab and Moku:Go.
% force_connect will overtake an existing connection
m = MokuMultiInstrument('192.168.###.###', 4, force_connect=true);

try
    %% Configure the instruments
    % Set the instruments and upload Cloud Compile bitstreams from your device
    % to your Moku
    bitstream = 'path/to/project/arithmetic/unzipped_bitstream';
    mcc = m.set_instrument(1, @MokuCloudCompile, bitstream);
    osc = m.set_instrument(2, @MokuOscilloscope);

    % configure routing
    connections = [struct('source','Slot1OutA', 'destination','Slot2InA');
                   struct('source','Slot2OutB', 'destination','Slot2InB');
                   struct('source','Slot2OutA', 'destination','Slot1InA');
                   struct('source','Slot2OutB', 'destination','Slot1InB')];

    m.set_connections(connections);

    %% Configure the Oscilloscope to generate a ramp wave and square wave with
    % equal frequencies, then sync the phases
    osc.generate_waveform(1, 'Square', 'amplitude',50e-3, 'frequency',1e3, ...
        'duty',50);
    osc.generate_waveform(2, 'Ramp', 'amplitude',50e-3, 'frequency',1e3, ...
        'symmetry',50);
    osc.sync_output_phase();

    % Set the time span to cover four cycles of the waveforms
    osc.set_timebase(-2e-3, 2e-3);
    osc.set_trigger('type','Edge', 'edge','Rising', 'level',0, 'mode','Normal', ...
        'source','ChannelB', 'auto_sensitivity',false, 'hysteresis',10e-3);

    %% Set up plots
    figure

    % Set Control Register 1 to choose to add (0b00), subtract (0b01) or
    % multiply (0b10) the input signals
    mcc.set_control(1, 0b00);
    % Retrieve the data
    data = osc.get_data('wait_reacquire',true);
    % Plot the result and configure labels for the axes
    add_graph = subplot(3,1,1);
    plot(add_graph, data.time, data.ch1, 'LineWidth', 1.5);
    title(add_graph, 'Add - 0b00');
    ylabel(add_graph,'Amplitude (V)');
    grid on;
    axis tight;

    % Repeat these steps for each option
    mcc.set_control(1, 0b01);
    data = osc.get_data('wait_reacquire',true);
    subtract_graph = subplot(3,1,2);
    plot(subtract_graph, data.time, data.ch1, 'LineWidth', 1.5);
    title(subtract_graph,'Subtract - 0b01');
    ylabel(subtract_graph,'Amplitude (V)');
    grid on;
    axis tight;

    mcc.set_control(1, 0b10);
    data = osc.get_data('wait_reacquire',true);
    multiply_graph = subplot(3,1,3);
    plot(multiply_graph, data.time, data.ch1, 'LineWidth', 1.5);
    title(multiply_graph, 'Multiply - 0b10');
    ylabel(multiply_graph,'Amplitude (V)');
    xlabel(multiply_graph,'Time (s)');
    grid on;
    axis tight;

    legend

catch ME
    % End the current connection session with your Moku
    m.relinquish_ownership();
    rethrow(ME)
end

m.relinquish_ownership();
