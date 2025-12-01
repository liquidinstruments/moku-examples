LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;

ARCHITECTURE Behavioural OF CustomInstrument IS
  -- Component Declarations
  COMPONENT MovingAverage 
  PORT( Clk                               :   IN    std_logic;
        Reset                             :   IN    std_logic;
        InputA                            :   IN    signed(15 DOWNTO 0);
        InputB                            :   IN    signed(15 DOWNTO 0);
        Control0                          :   IN    signed(15 DOWNTO 0);
        Control1                          :   IN    signed(15 DOWNTO 0);
        OutputA                           :   OUT   signed(15 DOWNTO 0);
        OutputB                           :   OUT   signed(15 DOWNTO 0)
      );
  END COMPONENT;
  COMPONENT MovingMedian
  PORT ( Clk                     : in  std_logic;
         Reset                   : in  std_logic;
         Input                   : in  signed(15 downto 0);
         Output                  : out signed(15 downto 0)
       );
  END COMPONENT;
  
BEGIN
  u_MovingAverage : MovingAverage
    PORT MAP( Clk      => Clk,
              Reset    => Reset,
              InputA   => InputA(11 downto 0) & "0000",
              InputB   => InputB, 
              Control0 => signed(Control(0)(15 downto 0)),
              Control1 => signed(Control(1)(15 downto 0)),
              OutputA  => OutputA,
              OutputB  => open
            );

  u_MovingMedian : MovingMedian
    PORT MAP( Clk    => Clk,
              Reset  => Reset,
              Input  => InputA(11 downto 0) & "0000",
              Output => OutputB
            );

END Behavioural;
