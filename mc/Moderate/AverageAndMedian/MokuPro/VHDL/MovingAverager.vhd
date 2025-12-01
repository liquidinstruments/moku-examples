library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity MovingAverage is
generic (
  G_AVERAGE_LENGTH_LOG         : integer := 8 );
port (
  Clk                      : in  std_logic;
  Reset                    : in  std_logic;
  InputA                   : in  signed(15 downto 0);
  InputB                   : in  signed(15 downto 0);
  Control0                 : in  signed(15 downto 0);
  Control1                 : in  signed(15 downto 0);
  OutputA                  : out signed(15 downto 0);
  OutputB                  : out signed(15 downto 0));
end MovingAverage;

architecture rtl of MovingAverage is

type t_moving_average is array (0 to 2**G_AVERAGE_LENGTH_LOG-1) of signed(15 downto 0);

signal p_moving_average                 : t_moving_average;
signal r_acc                            : signed(16+G_AVERAGE_LENGTH_LOG-1 downto 0);  -- average accumulator
signal r_data_valid                     : std_logic;

begin

average : process(Clk, Reset)
begin
  if(Reset = '1') then
    r_acc              <= (others=>'0');
    p_moving_average   <= (others=>(others=>'0'));
    OutputA            <= (others => '0');
  elsif(rising_edge(Clk)) then
    p_moving_average   <= InputA & p_moving_average(0 to p_moving_average'length-2);
    r_acc              <= r_acc + InputA-p_moving_average(p_moving_average'length-1);
    OutputA              <= r_acc(16+G_AVERAGE_LENGTH_LOG-1 downto G_AVERAGE_LENGTH_LOG);  -- divide by 2^G_AVG_LEN_LOG  
  end if;
end process average;

end rtl;