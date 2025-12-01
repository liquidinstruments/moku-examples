ARCHITECTURE HDLCoderWrapper OF CustomWrapper IS
-- SIGNAL Declarations
SIGNAL ConstantHigh : std_logic            := '1'; 
 -- Component Declarations
  COMPONENT DSP
  PORT( Clk                                :   IN    std_logic;
        Reset                              :   IN    std_logic;
        clk_enable                         :   IN    std_logic;
        InputA                             :   IN    signed(15 DOWNTO 0);  -- sfix16_En0
        InputB                             :   IN    signed(15 DOWNTO 0);  -- sfix16_En0
        ce_out_0                           :   OUT   std_logic;
        ce_out_1                           :   OUT   std_logic;
        OutputA                            :   OUT   signed(15 DOWNTO 0);  -- sfix16_En0
        OutputB                            :   OUT   signed(15 DOWNTO 0)  -- sfix16_En0
        );  
  END COMPONENT;

BEGIN
  u_DSP : DSP
    PORT MAP( Clk => Clk,
                        Reset => Reset,
                        clk_enable => ConstantHigh,
                        InputA => InputA,
                        InputB => InputB,
                        ce_out_0 => open,
                        ce_out_1 => open,
                        OutputA => OutputA,
                        OutputB => OutputB             
            );
END HDLCoderWrapper; 
