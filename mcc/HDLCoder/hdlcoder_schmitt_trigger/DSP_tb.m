% Load the input data.
load('tb_data');
num_samples = length(input_signal);  % Get the dataset length.
outdata = zeros(num_samples,1);  % Initialize the output array.

for n = 1:num_samples
  % Call DSP function.
  outdata(n) = DSP(input_signal(n)*2^16);  % Run the simulation.
end
 
% Visualize the results.
yyaxis left
plot(input_signal);
yyaxis right
plot(outdata);
legend('Input','Output');
xlim([0,4e3]);
