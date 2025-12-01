library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;


-- Designed by Brian J. Neff / Liquid Instruments
-- Will use the devices internal clock along with Control Register inputs to create a variable frequency and pulse width output
-- This code will work as is on any version of the Moku.  However, there is an alternate version that also uses the DIO for the Moku:Go
-- Moku should be configured as follows:
-- Control0 register must be non-zero integer
-- Control1 register must be non-zero integer 

architecture MaskWrapper of CustomWrapper is
  begin
    U_PulseMask: entity WORK.PulseMask
      port map(
          clk => Clk,
          passthrough => InputB, -- Will pass this signal through to output when mask is high
          divider => unsigned(Control0(31 downto 0)), -- Output pulse divider to control frequency
          duty => unsigned(Control1(31 downto 0)), -- Sets the duty cycle of the output pulse
          finalOut => OutputA, -- Either 0 (when Mask is Low) or InputB (when Mask is High)
          maskDAC => OutputB -- Mask representation output to DAC linked to OutputC in Multi-instrument Mode
      );  
      
end architecture;
