library IEEE;
use IEEE.Std_Logic_1164.All;
use IEEE.Numeric_Std.all;

entity TicksCounter is
    generic(  
        counterWidth : integer := 20
    );
    port (
        clk : in std_logic;
        reset : in std_logic;
        enable : in std_logic;
        encoderChA : in std_logic;
        encoderChB : in std_logic;
        leadChA : out std_logic;
        counts : out signed(counterWidth-1 downto 0)
        -- edges: out signed(15 downto 0) -- Debug edge counts output
    );
end entity;

architecture Behavioral of TicksCounter is
    signal posEdgeChA : std_logic;
    signal negEdgeChA : std_logic;
    signal posEdgeChB : std_logic;
    signal negEdgeChB : std_logic;

    signal posEdgeChADly : std_logic;
    signal negEdgeChADly : std_logic;
    signal posEdgeChBDly : std_logic;
    signal negEdgeChBDly : std_logic;

    signal prevChA : std_logic;
    signal prevChB : std_logic;
begin

    -- Edges direction detection
    posEdgeChA <= (not prevChA) and encoderChA; -- ChA positive edge detection
    negEdgeChA <= prevChA and (not encoderChA); -- ChA negative edge detection

    posEdgeChB <= (not prevChB) and encoderChB; -- ChB positive edge detection
    negEdgeChB <= prevChB and (not encoderChB); -- ChB negative edge detection

    process(clk) is
    begin

        if rising_edge(clk) then
          if reset then
            leadChA <= '0';
          
          -- Check leading encoder channel
          elsif enable then

            -- ChA Leading: ChA rising when ChB is low
            if posEdgeChA and (not encoderChB) then
              leadChA <= '1';

            -- ChB Leading: ChB rising when ChA is low
            elsif posEdgeChB and (not encoderChA) then
              leadChA <= '0';
            
            -- Keep the last valid value
            else
              leadChA <= leadChA;
            end if;
          
          -- Reset leadChA when counter is disabled
          else
            leadChA <= '0';
          end if;
        end if;

    end process;
    

    process(clk) is
    begin

        if rising_edge(Clk) then

          -- Encoder channels buffer for edge detection
          prevChA <= encoderChA;
          prevChB <= encoderChB;
          
          -- Compensate for leadChA delay
          -- leadChA is one clock cycle later than edge detector
          posEdgeChADly <= posEdgeChA;
          negEdgeChADly <= negEdgeChA;
          posEdgeChBDly <= posEdgeChB;
          negEdgeChBDly <= negEdgeChB;
        end if;
        
    end process; 


    process(clk) is
    begin

        if rising_edge(Clk) then
          if reset then
            counts <= (others => '0');
          
          -- Start counting
          elsif enable = '1' then

            -- Running forwards when ChA is leading
            if (posEdgeChADly or negEdgeChADly or posEdgeChBDly or negEdgeChBDly) and leadChA then
              counts <= counts + to_signed(1,counterWidth);

            -- Running backwards when ChB is leading
            elsif (posEdgeChADly or negEdgeChADly or posEdgeChBDly or negEdgeChBDly) and (not leadChA) then
              counts <= counts - to_signed(1,counterWidth);
            
            -- Keep last counter value
            else
              counts <= counts;
            end if;
          
          -- Reset counter when coutner is disabled
          else
            counts <= (others => '0');
          end if;
        end if;

    end process;    
          
end architecture;
