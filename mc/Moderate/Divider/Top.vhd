-----------------------------------------------------
--
-- Top
-- Overall architecture wrapper for the DSP components
-- of the divider, providing the internal signals and
-- ports of the DSP block 
--
-----------------------------------------------------

ARCHITECTURE HDLCoderWrapper OF CustomInstrument IS
-- SIGNAL Declarations
SIGNAL ConstantHigh:std_logic  :='1';
--Component Declarations
COMPONENT DSP
  PORT( clk                               :   IN    std_logic;
        reset                             :   IN    std_logic;
        InputA                            :   IN    signed(15 DOWNTO 0);  -- int16
        InputB                            :   IN    signed(15 DOWNTO 0);  -- int16
        Control0                          :   IN    std_logic_vector(15 DOWNTO 0);  -- int16
        OutputA                           :   OUT   signed(15 DOWNTO 0)  -- uint16
        );
END COMPONENT;


BEGIN
  u_DSP:DSP
  PORT MAP(clk => clk,
           reset => reset,
           InputA=>InputA,
           InputB=>InputB,
           Control0=>Control(0)(15 downto 0),
           OutputA=>OutputA
           );
END HDLCoderWrapper;