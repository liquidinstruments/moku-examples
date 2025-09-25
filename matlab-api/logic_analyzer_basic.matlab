%% Basic LogicAnalyzer Example
%
%  This example demonstrates how you can configure the LogicAnalyzer instrument.
%
%  (c) Liquid Instruments Pty. Ltd.
%

%% Connect to your Moku
% Connect to your Moku and deploy the Logic Analyzer instrument
% force_connect will overtake an existing connection
m = MokuLogicAnalyzer('192.168.###.###', force_connect=true);

try
    patterns = [struct('pin', 1, 'pattern', repmat([1],1,1024)), ...
        struct('pin', 2, 'pattern', repmat([0],1,1024)), ...
        struct('pin', 3, 'pattern', repmat([1 0],1,512)), ...
        struct('pin', 4, 'pattern', repmat([0 1],1,512))];
    
    
    m.set_pattern_generator(1, patterns, 'divider', 8);
    
    m.set_pin_mode(1, "PG1");
    m.set_pin_mode(2, "PG1");
    m.set_pin_mode(3, "PG1");
    m.set_pin_mode(4, "PG1");
    
    data = m.get_data('wait_reacquire', true, 'include_pins', [1, 2, 3, 4]);
    
    tiledlayout(4,1)
    
    ax1 = nexttile;
    p1 = stairs(data.time, data.pin1, 'Color','r');
    
    ax2 = nexttile;
    p2 = stairs(data.time, data.pin2, 'Color','r');
    
    ax3 = nexttile;
    p3 = stairs(data.time, data.pin3, 'Color','r');
    
    ax4 = nexttile;
    p4 = stairs(data.time, data.pin4, 'Color','r');
    

    
    linkaxes([ax1 ax2 ax3 ax4],'xy');
    
    %% Receive and plot new data frames
    while 1
        data = m.get_data();
        set(p1,'XData',data.time,'YData',data.pin1);
        set(p2,'XData',data.time,'YData',data.pin2);
        set(p3,'XData',data.time,'YData',data.pin3);
        set(p4,'XData',data.time,'YData',data.pin4);
        
        axis tight
        pause(0.1)
    end
    
catch ME
    % End the current connection session with your Moku
    m.relinquish_ownership();
    rethrow(ME)
end


% End the current connection session with your Moku
m.relinquish_ownership();