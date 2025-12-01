function OutputA = DSP(InputA)
% #codegen
persistent out0;  % When converted to HDL, this will mean out0 is stored 
                  % between successive evaluations of the logic below, 
                  % i.e. is registered.

if isempty(out0)  % Initializes the value once. As HDL, this initialization 
    out0 = 0;     % happens when the design is reset.
end

upperThreshold = floor(2^15/10);  % Set the upper threshold to 1/10 of the 
                                  % full positive range.
lowerThreshold = -floor(2^15/10);  % Set the lower threshold to 1/10 of the 
                                   % full negative range.
 
if InputA >upperThreshold  % Logic to perform the Schmitt trigger function.
    out0 = 2^15-1;
elseif InputA <lowerThreshold
    out0 = 0;
end

OutputA = out0;  % Assign variable out0 to the output.

