%% Basic Phasemeter Example
%
%  This example demonstrates how you can configure the Phasemeter
%  instrument to measure 4 independent signals.
%
%  (c) Liquid Instruments Pty. Ltd.
%

%% Connect to your Moku
% Connect to your Moku and deploy the Phasemeter instrument
% force_connect will overtake an existing connection
m = MokuPhasemeter('192.168.###.###', force_connect=true);

try
    % Set frontend of all input channels to 50 Ohm, DC coupled, 4 Vpp
    % range
    m.set_frontend(1,'50Ohm','DC','4Vpp');
    m.set_frontend(2,'50Ohm','DC','4Vpp');
    m.set_frontend(3,'50Ohm','DC','4Vpp');
    m.set_frontend(4,'50Ohm','DC','4Vpp');
    
    % Configure Output channel 1 to generate sine waves at 1 Vpp, 2 MHz
    m.generate_output(1, 'Sine', 'amplitude',1, 'frequency',2e6);
    % Configure Output channel 2 to be phase locked to Input 2 signal at an
    % amplitude of 0.5 Vpp
    m.generate_output(2, 'Sine', 'amplitude',0.5, 'phase_locked',true);
    % Configure Output channel 3 and 4 to generate measured phase at a
    % scaling of 1 V/cycle and 10 V/cycle respectively
    m.generate_output(3, 'Phase','scaling',1);
    m.generate_output(4, 'Phase','scaling',10);
    
    % Set the acquisition speed to 596Hz for all channels
    m.set_acquisition_speed('596Hz');
    
    % Set all input channels to 2 MHz, bandwidth 100 Hz
    m.set_pm_loop(1,'auto_acquire',false,'frequency',2e6,'bandwidth','100Hz');
    m.set_pm_loop(2,'auto_acquire',false,'frequency',2e6,'bandwidth','100Hz');
    m.set_pm_loop(3,'auto_acquire',false,'frequency',2e6,'bandwidth','100Hz');
    m.set_pm_loop(4,'auto_acquire',false,'frequency',2e6,'bandwidth','100Hz');
    
    % Get all the data available from the Moku
    data = m.get_data();
    
catch ME
    % End the current connection session with your Moku
    m.relinquish_ownership();
    rethrow(ME)
end

m.relinquish_ownership();