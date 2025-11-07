library ieee;

architecture Behavioural of CustomWrapper is

  signal s_temp1 : std_logic_vector(15 downto 0);
  signal s_temp2 : std_logic_vector(15 downto 0);  

component Beamsplitter
    port (
        clk       : in  std_logic;
        rst       : in  std_logic;
        signal_in : in  std_logic_vector(15 downto 0);
        noise_input : in  std_logic_vector(15 downto 0);
        out0      : out std_logic_vector(15 downto 0);
        out1      : out std_logic_vector(15 downto 0)
    );
end component;
  
begin
  
  BeamSplit : Beamsplitter
    port map (
      clk       => clk,
      rst       => Reset,
      signal_in => std_logic_vector(InputA),
      noise_input => std_logic_vector(InputB),
      out0      => s_temp1,
      out1      => s_temp2
    );
  
  OutputA <= signed(s_temp1);
  OutputB <= signed(s_temp2);    

end architecture;
