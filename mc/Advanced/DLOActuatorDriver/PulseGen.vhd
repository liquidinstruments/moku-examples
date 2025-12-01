library IEEE;
use IEEE.Std_Logic_1164.All;
use IEEE.Numeric_Std.all;

entity PulseGen is
  port (
      clk : in std_logic;
      reset : in std_logic;
      enable : in std_logic;
      deadInterval : in unsigned(15 downto 0);
      activeInterval : in unsigned(15 downto 0);
      pulseOut : out std_logic
  );
end entity;

architecture Behavioral of PulseGen is
  signal counter : unsigned(deadInterval'length downto 0);
begin  
  process(clk) is
  begin
  
      if rising_edge(Clk) then
        if reset then
          pulseOut <= '0';
          counter <= (others => '0');

        -- Start pulse generation
        elsif enable then

          -- Start output pulse after dead interval
          if deadInterval = counter and pulseOut = '0' then
            pulseOut <= '1';

            -- Reset counter
            counter <= (others => '0');
          
          -- Stop output pulse after active interval
          elsif activeInterval = counter and pulseOut = '1' then          
            pulseOut <= '0';   

            -- Reset counter
            counter <= (others => '0');     

          -- Stay in the same state                                       
          else
            pulseOut <= pulseOut;

            -- Increment counter
            counter <= counter + to_unsigned(1,counter'length);
          end if;

        -- Disable pulse generation
        else
           pulseOut <= '0';
           counter <= (others => '0');                                                      
        end if;
      end if;

  end process;
end architecture;                                                               
