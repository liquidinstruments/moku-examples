library IEEE;
use IEEE.Std_Logic_1164.all;
use IEEE.Numeric_Std.all;

-- Designed by Brian J. Neff / Liquid Instruments
-- Will produce a clock divider and output the divided clock to DIO Pin 9
-- Moku:Go should be configured as follows:
-- DIO Pin 0 to Input - Will reset the system on logical True
-- DIO Pin 8 to Output - Will output the divided clock pulse 
-- All other pins remain unused and can be configured as input or output

architecture ClkMeasureWrapper of CustomWrapper is

  begin
    U_ClkMeas: entity WORK.ClkMeas
      port map(
          clk => Clk,
          reset => InputA(0),
          pulse => OutputA(8)
    );  
end architecture;
