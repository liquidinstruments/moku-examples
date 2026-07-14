%% GenInst Example
%
% This example demonstrates how you can configure GenInst using
% Multi-Instrument Mode and output the result to the Oscilloscope.
% This example is written for Moku:Pro and can be changed for
% other devices.
%  (c) Liquid Instruments Pty Ltd
%


%% Connect to your Moku
% Connect to Moku via its IP address. Change platform_id to:
%   - 3, 5 or 8 for Moku:Delta, or
%   - 4 for Moku:Pro, or
%   - 2 or 3 for Moku:Lab and Moku:Go.
% force_connect will overtake an existing connection
m = MokuMultiInstrument('192.168.###.###', 4, force_connect=true);

try
    %% Configure the instruments
    % Set the instruments and upload GenInst bitstreams from your device
    % to your Moku
    bitstream = 'path/to/GenInst/bitstream';
    gi = m.set_instrument(1, @MokuGenInst, bitstream);
    osc = m.set_instrument(2, @MokuOscilloscope);

    % Configure routing
    connections = [struct('source','Slot1OutA', 'destination','Slot2InA');
                   struct('source','Slot1OutB', 'destination','Slot2InB');
                   struct('source','Slot2OutA', 'destination','Slot1InA');
                   struct('source','Slot2OutB', 'destination','Slot1InB')];
    m.set_connections(connections);

    % Set Control Registers as per the GenInst design plan
    % Setting a single control register
    gi.set_control(1, 0b00);
    
    %  Setting multiple control registers
    controls = struct( ...
      'idx', {0, 1, 2, 3, 4}, ...
      'value', {1e4, 23, 64, 55, 2147450879});
    gi.set_controls(controls);

    %% Configure the Oscilloscope to generate different waveforms
    osc.generate_waveform(1, 'Sine', 'amplitude',1, 'frequency',1e3);
    osc.generate_waveform(2, 'Sine', 'amplitude',0.1, 'frequency',500);
    
    % Sync the phase between the waveforms
    osc.sync_output_phase();

    % Set the time span to cover four cycles of the waveforms
    osc.set_timebase(-2e-3, 2e-3);

    %% Plot the acquired data and set up plotting parameters
    % Get initial data to set up plots
    data = osc.get_data('wait_complete', true);

    % Set up the plots
    figure
    lh = plot(data.time, data.ch1, data.time, data.ch2);
    xlabel(gca,'Time (sec)')
    ylabel(gca,'Amplitude (V)')

    %% Receive and plot new data frames
    while 1
        data = osc.get_data();
        set(lh(1),'XData',data.time,'YData',data.ch1);
        set(lh(2),'XData',data.time,'YData',data.ch2);

        axis tight
        pause(0.1)
    end
    
catch ME
    % End the current connection session with your Moku
    m.relinquish_ownership();
    rethrow(ME)
end

m.relinquish_ownership();
