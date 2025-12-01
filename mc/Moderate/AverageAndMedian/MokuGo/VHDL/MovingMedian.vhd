library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity MovingMedian is
    port (
          Clk                     : in  std_logic;
          Reset                   : in  std_logic;
          Input                   : in  signed(15 downto 0);
          Output                  : out signed(15 downto 0)
    );
end entity;

architecture rtl of MovingMedian is
    type   time_window is array (0 to 4) of signed(15 downto 0);
    signal moving_window : time_window;
    signal sorted_window : time_window;
    signal median        : signed(15 downto 0);

begin

data_pipe : process(Clk)
  begin
    if rising_edge(Clk) then
      if (reset = '1') then
         moving_window <= (others => (others => '0'));
      else
         moving_window  <= Input & moving_window(0 to moving_window'length-2);
      end if;
    end if;
  end process;
  
sort_window:
    process (Clk)
      variable temp:      signed(15 downto 0);
      variable var_array: time_window;       
    begin
        var_array := moving_window;
        if rising_edge(clk) then
          if (reset = '1') then
            Sorted_window <= (others => (others => '0'));
            median <= (others => '0');
          else                                       
            for j in time_window'LEFT to time_window'RIGHT - 1 loop 
                for i in time_window'LEFT to time_window'RIGHT - 1 - j loop 
                    if var_array(i) > var_array(i + 1) then
                        temp := var_array(i);
                        var_array(i) := var_array(i + 1);
                        var_array(i + 1) := temp;
                    end if;
                end loop;
            end loop;
            sorted_window <= var_array;
            median <= sorted_window(2);
          end if; 
        end if;
    end process;
                                                 
Output <= median;
                                                 
end architecture rtl;
