%% Post-processing measured microvolt signals
%
%  This script post-processes the demodulated signals from the Jupyter
%  notebook in the same Git folder as this script. The demodulated signals
%  contain 10 cycles of a step function that corresponds to Morse
%  characters (dot: 0V, dash: 0.25V, space: -0.25V).
%  The script averages the 10 cycles of the demodulated Morse pattern,
%  divides each cycle into 14 segments (14 Morse characters), and
%  plots the average error of the middle value in each segment. The script
%  also outputs the interpretted Morse code of each cycle.
%
%  For an in-depth explanation of how to measure microvolt signals on 
%  Moku:Pro, please visit the application notes section on our website.
%
%  (c) Liquid Instruments Pty. Ltd.
%


% Clear all variables, close all figures, and clear the command window
clear all
close all
clc

% Load data from folder with matfiles
folder = 'C:\Users\Your\Folder\Path\Here';
files = dir(fullfile(folder, '*.mat'));

% Preallocate results arrays
error = zeros(1, length(files));
amp = zeros(1, length(files));
all_voltages = [];
all_times = [];

% Loop through each file in data folder
for k = 1:length(files)
    filename = files(k).name;

    % Get amplitude from parameters in filename
    tokens = regexp(filename, '(\d+)dB_(\d+)mVpp', 'tokens');
    numbers = str2double(tokens{1});
    dB = numbers(1);
    mVpp = numbers(2);
    amp(k) = (mVpp./1000).*10.^(-dB./20);

    % Call the processing function, passing folder and filename
    [error(k), time_cycle, voltage_cycle] = process_PM_data(folder, filename);
    
    % Store time values for initial message in measurements for plotting
    if k == 1
        all_times = time_cycle;
    end
    
    % Save first message voltages for plotting
    all_voltages(k,:) = voltage_cycle;
end

% Sort amplitudes and get the sorting indices
[sortedamp, idx] = sort(amp);

% Use the indices to reorder corresponding error values
sortederror = error(idx);
sortedvoltages = all_voltages(idx,:);
legend_strings = string(sortedamp.*1e6) + " µV";

% Plot first message of each amplitude measurement
figure(1)
subplot(2,1,1)
cmap = parula(length(files)); 
hold on
for k = 1:length(files)
    plot(all_times,sortedvoltages(k,:),'LineWidth',2,'Color',cmap(k,:))
end
xlabel('Time (s)')
ylabel('Voltage (V)')
axis tight
legend(legend_strings(1:k),'Location','eastoutside')
set(gca,'FontSize',20)

% Plot first message of measurements with amplitudes < 10 microvolts
% The data in Figure 7a and Figure 9 contains 4 datasets with amplitudes 
% ≥ 10 microvolts. 
subplot(2,1,2)
cmap = parula(length(files)); 
hold on
for k = 1:length(files)-4
    plot(all_times,sortedvoltages(k,:),'LineWidth',2,'Color',cmap(k,:))
end
xlabel('Time (s)')
ylabel('Voltage (V)')
axis tight
legend(legend_strings(1:k),'Location','eastoutside')
set(gca,'FontSize',20)

% Plot averaged error for each amplitude
figure
scatter(sortedamp.*1e6,sortederror,200,cmap,'^','filled','MarkerEdgeColor',"#000000") 
xlabel('Amplitude (µV)', 'FontName', 'Helvetica', 'FontSize', 14);
ylabel('Averaged error (%)')
colormap(gca,"parula")
axis tight
set(gca, 'XScale', 'log')
set(gca,'FontSize',20)
grid on

function [overall_avg_error, first_time, first_voltage] = process_PM_data(folder,filename)

    % Extract time and voltage arrays in the data file
    file = fullfile(folder, filename);
    contents = load(file);
    data = contents.moku.data;
    time_full = data(:,1);
    voltage_full = data(:,2);
    
    start_idx = 4330;                   % Approximate starting idx from plots
    start_time = time_full(start_idx);  % Adjust as needed
    cycle_duration = 33.33;             % Period based on AWG waveform frequency
    
    mid = (max(voltage_full(start_idx:end))-min(voltage_full(start_idx:end)))/2;    % Calculate half of the voltage range
    mid_y = max(voltage_full(start_idx:end))-mid;                                   % Calculate midpoint of voltage range
    offset = mid_y - 0;                                                             % Adjust if there is an unwanted offset
    
    % Preallocate final message
    final_message = '';
    
    all_segment_mids = [];  % Create empty matrix: rows = cycles, cols = Morse segments
    count = 0;              % Create count variable to loop through each Morse cycle
    
    % Loop through each 33.33-second cycle
    while start_time + cycle_duration <= time_full(end) && count < 10
        count = count + 1;

        % Define time window for this cycle
        end_time = start_time + cycle_duration;
        indices = (time_full >= start_time) & (time_full <= end_time);
    
        % Extract time and voltage for this cycle
        time = time_full(indices);
        voltage = voltage_full(indices)-offset;

        % Define the first time and voltage of the data to be divided into 10 cycles 
        % This slices out the data prior to when the first cycle begins
        if count == 1
            first_time = time-time(1);
            first_voltage = voltage;
        end
    
        % Divide the amplitude range into thirds to set threshold values
        % for each Morse character
        range = 0.5;                            % The demodulated waveform has a peak to peak amplitude of 0.5V
        bottom_third = -0.25 + (range*0.333);   % Get the value that marks the bottom third of the 0.5V range
        top_third = 0.25 - (range*0.333);       % Get the value that marks the top third of the 0.5V range

        num_segments = 14; % Number of Morse characters (segments)

        % Define the expected value of the voltage in the middle of each
        % segment for dot, dash, and space
        top = 0.25;                             % Voltage for dash is 0.25V
        mid = 0;                                % Voltage for dot is 0V
        bottom = -0.25;                         % Voltage of space is -0.25V

        % Make list of encoded Morse levels for expected message "Moku": 
        % -- --- -.- ..-
        segment_refs = [top, top, bottom, top, top, top, bottom, top, ...
                          mid, top, bottom, mid, mid, top]; 

        % Calculate how many points will be in each Morse segment
        num_points = length(voltage);
        points_per_segment = floor(num_points / num_segments);
        segment_middle_values = zeros(1, num_segments);
    
        % Find the middle value of each segment
        for i = 1:num_segments
            start_seg = (i - 1) * points_per_segment + 1;
            end_seg = i * points_per_segment;
            if i == num_segments
                end_seg = num_points;
            end
            mid_idx = round((start_seg + end_seg) / 2);
            segment_middle_values(i) = voltage(mid_idx);
            if i == 14
                all_segment_mids = [all_segment_mids; segment_middle_values];  % Append as row
            end
        end
    
        % Decode Morse-like symbols using threshold values
        message = '';
        for j = 1:length(segment_middle_values)
            if segment_middle_values(j) > top_third
                message = [message, '-'];
            elseif segment_middle_values(j) < bottom_third
                message = [message, ' '];
            else
                message = [message, '.'];
            end
        end
    
        % Print the decoded Morse message of each segment
        fprintf('Cycle starting at %.2f s — Morse Code Message: %s\n', start_time, message);
    
        % Move to the next cycle
        start_time = end_time;
    end
    
    % Compute mean and std per segment as before
    segment_means = mean(all_segment_mids, 1);
    segment_std = std(all_segment_mids, 0, 1);  % std deviation (N-1)
    
    % Compute absolute error per cycle and per segment
    abs_error_matrix = (abs(all_segment_mids - segment_refs)/segment_refs)*100;  % size: [cycles x segments]
    
    % Compute average error per segment
    segment_avg_errors = mean(abs_error_matrix, 1);  % size: [1 x 14]
    
    % Compute overall average error
    overall_avg_error = mean(abs_error_matrix(:));

end