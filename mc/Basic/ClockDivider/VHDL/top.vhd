library IEEE;
use IEEE.Std_Logic_1164.all;
use IEEE.Numeric_Std.all;

-- Designed by Brian J. Neff / Liquid Instruments
-- Will produce a clock divider and output the divided clock to specified pin
-- Moku:Go should be configured as follows:
-- DIO Pin 0 to Input - Will reset the system on logical True
-- DIO Pin 8 to Output - Will output the divided clock pulse by a factor of 2
-- DIO Pin 9 to Output - Will output the divided clock pulse by a factor of 4
-- DIO Pin 10 to Output - Will output the divided clock pulse by a factor of 6
-- All other pins remain unused and can be configured as input or output

architecture ClkDividerWrapper of CustomWrapper is

  begin
    U_ClkDivider1: entity WORK.ClkDivider
      generic map(
          divider => x"0001"
      )
      port map(
          clk => Clk,
          reset => InputA(0),
          pulse => OutputA(8)
    );

    -- Create additional entities to highlight value of using the generic 
    U_ClkDivider2: entity WORK.ClkDivider
      generic map(
          divider => x"0002"
      )
      port map(
          clk => Clk,
          reset => InputA(0),
          pulse => OutputA(9)
    );
    U_ClkDivider3: entity WORK.ClkDivider
      generic map(
          divider => x"0003"
      )
      port map(
          clk => Clk,
          reset => InputA(0),
          pulse => OutputA(10)
    );  
end architecture;
