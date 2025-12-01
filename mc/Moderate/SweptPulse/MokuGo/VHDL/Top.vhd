library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

-- Designed by Brian J. Neff / Liquid Instruments
-- Will use the devices internal clock to create a variable frequency and pulse width output
-- Will need to be adjusted for the clock rate of specific device (Moku:Go clock is 31.25 MHz)
-- Moku should be configured as follows:
-- DIO Pin 0 is input
-- DIO Pin 8 is output
-- Control0 register must be non-zero integer
-- Control1 register must be non-zero integer 

architecture MaskWrapper of CustomInstrument is
  begin
    U_PulseMask: entity WORK.PulseMask
      port map(
          clk => Clk,
          reset => InputA(0),                         -- Reset input on DIO pin-0
          passthrough => InputB,                      -- Will pass this signal through to output when mask is high
          divider => unsigned(Control(0)(31 downto 0)), -- Output pulse divider to control frequency
          duty => unsigned(Control(1)(31 downto 0)),    -- Sets the duty cycle of the output pulse
          finalOut => OutputB,                        -- Either 0 (when Mask is 0) or InputB (when Mask is 1)
          maskDAC => OutputC,                         -- Mask representation output to DAC linked to OutputC in Multi-instrument Mode
          maskDIO => OutputA(8)                       -- Mask representation output to DIO pin-8 
      );  
      
end architecture;
