%% Arithmetic Cloud Compile Example
%
%  This example demonstrates how you can configure Cloud Compile to chose to
% add, subtract or multiply two input signals using the control registers
% and output the result to the Oscilloscope.
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
    bitstream = 'path/to/project/arithmetic/bitstreams.tar';
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
    osc.generate_waveform(1, 'Square', 'amplitude',50e-3, 'frequency',1e3, ...
        'duty',50);
    osc.generate_waveform(2, 'Ramp', 'amplitude',50e-3, 'frequency',1e3, ...
        'symmetry',50);
    osc.sync_output_phase();

    % Set the time span to cover four cycles of the waveforms
    osc.set_timebase(-2e-3, 2e-3);
    osc.set_trigger('type','Edge', 'edge','Rising', 'level',0, 'mode','Normal', ...
        'source','ChannelB');

    %% Set up plots
    figure

    % Set Control Register 1 to choose to add (0b00), subtract (0b01) or
    % multiply (0b10) the input signals
    mcc.set_control(1, 0b00);
    % Retrieve the data
    data = osc.get_data('wait_reacquire',True);
    % Plot the result and configure labels for the axes
    add_graph = subplot(3,1,1);
    plot(add_graph, data.time, data.ch1);
    title(add_graph, 'Add - 0b00');
    ylabel(add_graph,'Amplitude (V)');

    % Repeat these steps for each option
    mcc.set_control(1, 0b01);
    data = osc.get_data('wait_reacquire',True);
    subtract_graph = subplot(3,1,2);
    plot(subtract_graph, data.time, data.ch1);
    title(subtract_graph,'Subtract - 0b01');
    ylabel(subtract_graph,'Amplitude (V)');

    mcc.set_control(1, 0b10);
    data = osc.get_data('wait_reacquire',True);
    multiply_graph = subplot(3,1,3);
    plot(multiply_graph, data.time, data.ch1);
    title(multiply_graph, 'Multiply - 0b10');
    xlabel(multiply_graph,'Time (s)');
    ylabel(multiply_graph,'Amplitude (V)');

    legend

catch ME
    % End the current connection session with your Moku
    m.relinquish_ownership();
    rethrow(ME)
end

m.relinquish_ownership();
