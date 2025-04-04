LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;

ARCHITECTURE SimulinkHDLCoderWrapper OF CustomWrapper IS
  -- Component Declarations
  COMPONENT DSP_fixpt
  PORT( Clk                               :   IN    std_logic;
        Reset                             :   IN    std_logic;
        InputA                            :   IN    signed(15 DOWNTO 0); 
        OutputA                           :   OUT   signed(15 DOWNTO 0)
        );
  END COMPONENT;

BEGIN
  u_DSP_fixpt : DSP_fixpt
    PORT MAP( Clk,
              Reset,
              InputA, 
              OutputA
            );

END SimulinkHDLCoderWrapper;