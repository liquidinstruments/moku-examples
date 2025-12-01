library IEEE;
use IEEE.Std_Logic_1164.All;
use IEEE.Numeric_Std.all;

entity DSP is
    generic(
        counterWidth : integer := 20
    );
    port (
        clk : in std_logic;
        reset : in std_logic;
        deadIntervalRun : in unsigned(15 downto 0);
        activeIntervalRun : in unsigned(15 downto 0);
        deadIntervalCorrect : in unsigned(15 downto 0);
        activeIntervalCorrect : in unsigned(15 downto 0);
        pulseNum : in signed(counterWidth-1  downto 0);
        overshootNum : in signed(counterWidth-1 downto 0);
        backPulseNum : in unsigned(counterWidth-1 downto 0);
        rangeTolerance : in unsigned(counterWidth-1 downto 0);
        waitInterval : in unsigned(31 downto 0);
        encoderChA : in std_logic;
        encoderChB : in std_logic;
        posLimit : in std_logic;
        negLimit : in std_logic;
        pinDirect : out std_logic;
        pulseOut : out std_logic;
        stateOut : out std_logic_vector(1 downto 0)
    );
end entity;

architecture Behavioral of DSP is  
  type t_State is (Waiting, Running, Correcting, Reversing);  
  signal State : t_State := Waiting;
  
  signal pulseEnable : std_logic;

  signal enableCounting : std_logic; 
  signal enableReverseCounting : std_logic;
  signal counts : signed(PulseNum'left downto 0);
  signal countsReverse : signed(PulseNum'left downto 0);
  signal countsWait : unsigned(waitInterval'left downto 0);                              
                                
  signal prevPulseNum :  signed(pulseNum'left downto 0);
  signal prevOvershootNum :  signed(overshootNum'left downto 0);                              
  signal prevBackPulseNum :  unsigned(backPulseNum'left downto 0);

  signal deadInterval : unsigned(deadIntervalRun'left downto 0);
  signal activeInterval : unsigned(activeIntervalRun'left downto 0);   

  signal pulseGenOut : std_logic;    

  signal dutyFlip : std_logic;                                 

begin
  -- Counting for normal runs                                
  TicksCounter: entity WORK.TicksCounter
  generic map(
      counterWidth => pulseNum'length
  )
  port map (
      clk => clk,
      reset => reset,
      enable => enableCounting,
      encoderChA => encoderChA,
      encoderChB => encoderChB,
      counts => counts
  );             
  -- Counting for reverse runs when limit switches are triggered
  TicksCounterReverse: entity WORK.TicksCounter
  generic map(
      counterWidth => pulseNum'length
  )
  port map (
      clk => clk,
      reset => reset,
      enable => enableReverseCounting,
      encoderChA => encoderChA,
      encoderChB => encoderChB,
      counts => countsReverse
  );                                  
  -- Drive pulse generator
  U_PulseGen: entity WORK.PulseGen
  port map(
      clk => clk,
      reset => reset,
      enable => pulseEnable,
      deadInterval => deadInterval,
      activeInterval => activeInterval,
      pulseOut => pulseGenOut
  );
  pulseOut <= (NOT pulseGenOut) when (dutyFlip = '1') else pulseGenOut;
                                   
  -- State machine: combine combinational and sequential logics together
  process(clk) is
  variable diffTarget : signed(pulseNum'left downto 0);   
  variable upperLimit : signed(pulseNum'left downto 0);     
  variable lowerLimit : signed(pulseNum'left downto 0);                                
  begin
      diffTarget := prevPulseNum - counts;
      upperLimit := prevPulseNum + signed(rangeTolerance);       
      lowerLimit := prevPulseNum - signed(rangeTolerance);                           
      if rising_edge(Clk) then
        if reset then
          prevPulseNum <= (others => '0');
          prevBackPulseNum <= (others => '0');                          
          countsWait <= (others => '0');              
          pulseEnable <= '0';
          enableCounting <= '0';
          enableReverseCounting <= '0';                          
          State <= Waiting;                         
        else
        case State is
          when Waiting =>                      
            -- Drive actuator away from positive/negative ends
            if posLimit = '1' or negLimit = '1' then
              State <= Reversing;
              pulseEnable <= '1'; 
              pinDirect <= NOT posLimit and negLimit; 
              dutyFlip <= NOT posLimit and negLimit; 
              enableCounting <= '0';
              enableReverseCounting <= '1';  

            -- Start driving when pulseNum changes                        
            elsif prevPulseNum = to_signed(0, prevPulseNum'length) and pulseNum /= to_signed(0, pulseNum'length) then   
              State <= Running;  
              pulseEnable <= '1';
              pinDirect <= NOT pulseNum(pulseNum'left);
              dutyFlip <= NOT pulseNum(pulseNum'left);
              enableCounting <= '1';
              enableReverseCounting <= '0';
            end if; 

            countsWait <= (others => '0');                              
            prevPulseNum <= pulseNum;
            prevOvershootNum <= overshootNum;                            
            prevBackPulseNum <= backPulseNum;   
            deadInterval <=  deadIntervalRun;
            activeInterval <= activeIntervalRun;  

            -- Debug state output
            stateOut <= "00";   
                              
          when Running =>
            -- Drive actuator away from positive/negative ends
            if posLimit = '1' or negLimit = '1' then
              pinDirect <= (NOT posLimit) and negLimit;  
              dutyFlip <= (NOT posLimit) and negLimit;                           
              State <= Reversing;
              enableCounting <= '0';
              enableReverseCounting <= '1';                          

            -- Stop driving when the pulse counts reach (target pulse number + overshoot number) [correct direction]
            -- or when absolute value of pulse counts is larger than (target pulse number + overshoot number) [wrong direction]
            elsif (prevPulseNum + prevOvershootNum) = counts or abs(prevPulseNum + prevOvershootNum) < abs(counts) then
              pulseEnable <= '0'; 
              enableCounting <= '1'; 
              enableReverseCounting <= '0';

              -- Wait until actuator is stable (not moving)
              -- after wait interval, start correcting
              if countsWait >= waitInterval then
                 State <= Correcting;

                 -- Change duty cycle
                 deadInterval <=  deadIntervalCorrect;
                 activeInterval <= activeIntervalCorrect;

                 -- Clear wait interval counter
                 countsWait <= to_unsigned(0, countsWait'length);   

              -- Stay in Running state until wait interval completes
              -- increment wait interval counter                    
              else
                 countsWait <= countsWait + to_unsigned(1, countsWait'length);
                 State <= Running;                                     
              end if;                                                                 
            end if; 

            -- Debug state output
            stateOut <= "01";

          when Correcting =>
            -- Drive actuator away from positive/negative ends
            if posLimit = '1' or negLimit = '1' then
              pinDirect <= (NOT posLimit) and negLimit;  
              dutyFlip  <= (NOT posLimit) and negLimit;                                           
              State <= Reversing;
              enableCounting <= '0';
              enableReverseCounting <= '1';      

            -- Correct actuator's overdrive
            -- and change driving direction 
            elsif counts > upperLimit or counts < lowerLimit then
              pulseEnable <= '1'; 
              enableCounting <= '1';                                       
              pinDirect <= NOT diffTarget(diffTarget'left);
              dutyFlip <= NOT diffTarget(diffTarget'left);                            
              countsWait <= (others => '0'); 

            -- Stop driving when the overdrive is compensated                   
            elsif counts < upperLimit and counts > lowerLimit then
              countsWait <= countsWait + to_unsigned(1, countsWait'length);
              pulseEnable <= '0'; 
              enableReverseCounting <= '0';

              -- Switch back to Waiting after Correcting state
              if countsWait >= waitInterval then
                 State <= Waiting;
                 enableCounting <= '0';      

              -- Stay in Correcting state until wait interval completes
              -- increment wait interval counter                                    
              else
                 State <= Correcting;
                 enableCounting <= '1';                                   
              end if;                   
            end if; 

            -- Debug state output
            stateOut <= "10";         

          when Reversing =>
            enableCounting <= '0';   

            -- Stop reversing after certain pulse counts                           
            if prevBackPulseNum <= unsigned(abs(countsReverse)) then
              enableReverseCounting <= '0';
              pulseEnable <= '0'; 
              State <=  Waiting;                            
            end if;

            -- Debug state output
            stateOut <= "11";    
                                     
          end case;                      
        end if;
      end if;
  end process;

end architecture;
