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
    type   time_window        is array (0 to 4) of signed(15 downto 0);
    type   staged_sort_window is array(0 to 5) of time_window;
    signal moving_window      : time_window;
    signal staged_sort        : staged_sort_window;
    signal median             : signed(15 downto 0);

begin

moving_time_window : process(Clk)
  begin
    if rising_edge(Clk) then
      if (reset = '1') then
         moving_window <= (others => (others => '0'));
      else
         moving_window  <= Input & moving_window(0 to moving_window'length-2);
      end if;
    end if;
  end process;
  
multi_compare_and_sort : process (Clk)
      variable temp       : signed(15 downto 0);
      variable sort_var   : time_window;
    begin
        if rising_edge(clk) then
          if (reset = '1') then
            staged_sort <= (others => (others => (others => '0')));                                             
            median <= (others => '0');
          else                                       

--------------------------------------------------------------------------------
--  Assigned first stage of the staged sort to the current moving time window
---------------------------------------------------------------------------------

                staged_sort(0) <= moving_window;

--------------------------------------------------------------------------------
--  Stage1 is top pair compare
---------------------------------------------------------------------------------

                if (staged_sort(0)(0) > staged_sort(0)(1)) then
                  staged_sort(1)(0) <= staged_sort(0)(1);
                  staged_sort(1)(1) <= staged_sort(0)(0);
                else
                  staged_sort(1)(0) <= staged_sort(0)(0);
                  staged_sort(1)(1) <= staged_sort(0)(1);
                end if;

                if (staged_sort(0)(2) > staged_sort(0)(3)) then
                  staged_sort(1)(2) <= staged_sort(0)(3);
                  staged_sort(1)(3) <= staged_sort(0)(2);
                else
                  staged_sort(1)(2) <= staged_sort(0)(2);
                  staged_sort(1)(3) <= staged_sort(0)(3);
                end if;

                staged_sort(1)(4) <= staged_sort(0)(4);


--------------------------------------------------------------------------------
--  Stage2 is bottom pair compare
---------------------------------------------------------------------------------
                staged_sort(2)(0) <= staged_sort(1)(0);

                if (staged_sort(1)(1) > staged_sort(1)(2)) then
                  staged_sort(2)(1) <= staged_sort(1)(2);
                  staged_sort(2)(2) <= staged_sort(1)(1);
                else
                  staged_sort(2)(1) <= staged_sort(1)(1);
                  staged_sort(2)(2) <= staged_sort(1)(2);
                end if;

                if (staged_sort(1)(3) > staged_sort(1)(4)) then
                  staged_sort(2)(3) <= staged_sort(1)(4);
                  staged_sort(2)(4) <= staged_sort(1)(3);
                else
                  staged_sort(2)(3) <= staged_sort(1)(3);
                  staged_sort(2)(4) <= staged_sort(1)(4);
                end if;


--------------------------------------------------------------------------------
--  Stage3 is top pair compare
---------------------------------------------------------------------------------

                if (staged_sort(2)(0) > staged_sort(2)(1)) then
                  staged_sort(3)(0) <= staged_sort(2)(1);
                  staged_sort(3)(1) <= staged_sort(2)(0);
                else
                  staged_sort(3)(0) <= staged_sort(2)(0);
                  staged_sort(3)(1) <= staged_sort(2)(1);
                end if;

                if (staged_sort(2)(2) > staged_sort(2)(3)) then
                  staged_sort(3)(2) <= staged_sort(2)(3);
                  staged_sort(3)(3) <= staged_sort(2)(2);
                else
                  staged_sort(3)(2) <= staged_sort(2)(2);
                  staged_sort(3)(3) <= staged_sort(2)(3);
                end if;

                staged_sort(3)(4) <= staged_sort(2)(4);
 

--------------------------------------------------------------------------------
--  Stage4 is bottom pair compare
---------------------------------------------------------------------------------
                staged_sort(4)(0) <= staged_sort(4)(0);

                if (staged_sort(3)(1) > staged_sort(3)(2)) then
                  staged_sort(4)(1) <= staged_sort(3)(2);
                  staged_sort(4)(2) <= staged_sort(3)(1);
                else
                  staged_sort(4)(1) <= staged_sort(3)(1);
                  staged_sort(4)(2) <= staged_sort(3)(2);
                end if;

                if (staged_sort(3)(3) > staged_sort(3)(4)) then
                  staged_sort(4)(3) <= staged_sort(3)(4);
                  staged_sort(4)(4) <= staged_sort(3)(3);
                else
                  staged_sort(4)(3) <= staged_sort(3)(3);
                  staged_sort(4)(4) <= staged_sort(3)(4);
                end if;

---------------------------------------------------------------------------------
--  Stage5 is top pair compare
---------------------------------------------------------------------------------

                if (staged_sort(4)(0) > staged_sort(4)(1)) then
                  staged_sort(5)(0) <= staged_sort(4)(1);
                  staged_sort(5)(1) <= staged_sort(4)(0);
                else
                  staged_sort(5)(0) <= staged_sort(4)(0);
                  staged_sort(5)(1) <= staged_sort(4)(1);
                end if;

                if (staged_sort(4)(2) > staged_sort(4)(3)) then
                  staged_sort(5)(2) <= staged_sort(4)(3);
                  staged_sort(5)(3) <= staged_sort(4)(2);
                else
                  staged_sort(5)(2) <= staged_sort(4)(2);
                  staged_sort(5)(3) <= staged_sort(4)(3);
                end if;

                staged_sort(5)(4) <= staged_sort(4)(4);
 

            median <= staged_sort(5)(2);
        end if;
      end if;
    end process;
                                                 
Output <= median;

end rtl;

