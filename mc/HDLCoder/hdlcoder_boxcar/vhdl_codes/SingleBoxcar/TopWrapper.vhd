LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;

ARCHITECTURE HDLCoderWrapper OF CustomWrapper IS
  -- Component Declarations
  
COMPONENT DSP -- Make sure this matches your VHDL code
PORT( Clk                               :   IN    std_logic;
      Reset                             :   IN    std_logic;
      InputA                            :   IN    signed(15 DOWNTO 0);  -- int16
      InputB                            :   IN    signed(15 DOWNTO 0);  -- int16
      TriggerLevel                      :   IN    signed(15 DOWNTO 0);  -- int16
      TriggerDelay                      :   IN    signed(15 DOWNTO 0);  -- int16
      GateWidth                         :   IN    signed(15 DOWNTO 0);  -- int16
      AvgLength                         :   IN    signed(15 DOWNTO 0);  -- int16
      SwitchControl                     :   IN    signed(15 DOWNTO 0);  -- int16
      Gain                              :   IN    signed(31 DOWNTO 0);  -- int32
      OutputA                           :   OUT   signed(15 DOWNTO 0);  -- int16
      OutputB                           :   OUT   signed(15 DOWNTO 0)  -- int16
);
END COMPONENT;

BEGIN
  Averager:DSP
  PORT MAP(Clk=>Clk,
           Reset=>Reset,
           InputA=>InputA,
           InputB=>InputB,
           
           TriggerLevel=>signed(Control0(15 downto 0)),
           TriggerDelay=>signed(Control1(15 downto 0)),
           GateWidth=>signed(Control2(15 downto 0)),
           AvgLength=>signed(Control3(15 downto 0)),
           SwitchControl=>signed(Control4(15 downto 0)),
           Gain=>signed(Control5(31 downto 0)),
           
           OutputA=>OutputA,
           OutputB=>OutputB
           );

END HDLCoderWrapper;