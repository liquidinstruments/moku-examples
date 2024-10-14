library IEEE;
use IEEE.Std_Logic_1164.All;
use IEEE.Numeric_Std.all;

architecture FloquetWrapper of CustomWrapper is
  constant counterWidth : integer := 32;
begin
  
  U_DSP: entity WORK.DSP
  generic map(
      counterWidth => counterWidth
  )
  port map(
      clk                     =>  clk,
      reset                   =>           Control0(0),
      deadIntervalRun         =>  unsigned(Control1(15 downto 0)),
      activeIntervalRun       =>  unsigned(Control1(31 downto 16)),
      deadIntervalCorrect     =>  unsigned(Control2(15 downto 0)),
      activeIntervalCorrect   =>  unsigned(Control2(31 downto 16)),
      pulseNum                =>    signed(Control3(counterWidth-1 downto 0)),
      overshootNum            =>    signed(Control4(counterWidth-1 downto 0)),
      backPulseNum            =>  unsigned(Control5(counterWidth-1 downto 0)),
      rangeTolerance          =>  unsigned(Control6(counterWidth-1 downto 0)),
      waitInterval            =>  unsigned(Control7(31 downto 0)),
      encoderChA              =>  InputA(0),
      encoderChB              =>  InputA(1),
      posLimit                =>  InputA(2),
      negLimit                =>  InputA(3),
      pinDirect               =>  OutputA(4),
      pulseOut                =>  OutputA(5),

      -- Debug state output ports
      stateOut(0)             =>  OutputA(10),
      stateOut(1)             =>  OutputA(11)
  );

  -- Route inputs back to outputs for debugging
  OutputA(6) <= InputA(0);
  OutputA(7) <= InputA(1);
  OutputA(8) <= InputA(2);
  OutputA(9) <= InputA(3);
  
end architecture;