Random multiplexer

This file takes two inputs, InputA and InputB. InputA is the signal and InputB is the gate voltage. 

Every N clock cycles the module reads the MSB (sign) of InputB and diverts to OutputA if positive and OutputB if negative. N is determined by the value of Control0 

InputB can be a gate, DC voltage, or random noise. For an example, see this application note:

https://liquidinstruments.com/application-notes/second-order-correlation-using-the-moku-time-frequency-analyzer/
