library IEEE;
use IEEE.Std_Logic_1164.all;
use IEEE.Numeric_Std.all;

entity BoxcarAverager is
    port(
        Clk : in std_logic;
        Reset : in std_logic;

        InputA : in signed(15 downto 0);
        InputB : in signed(15 downto 0);

        OutputA : out signed(15 downto 0);
        OutputB : out signed(15 downto 0)
    );
end BoxcarAverager;

architecture Behavioural of BoxcarAverager is
	constant LOG_GATE_LEN : integer := 3;
	constant LOG_AVG_LEN : integer := 3;

	type StorageArrayGate is array (0 to 2 ** LOG_GATE_LEN - 1) of signed(15 downto 0);
    type StorageArrayOut is array (0 to 2 ** LOG_AVG_LEN - 1) of signed(15 downto 0);
	type StateType is (WaitForTrigger, Triggered);

    signal State : StateType := WaitForTrigger;

    signal PreviousInputB : signed(15 downto 0) := (others => '0');

	signal EventDataArray : StorageArrayGate := (others => (others => '0'));
    signal EventSum : signed(16 + LOG_GATE_LEN - 1 downto 0) := (others => '0');
    signal EventCounter : integer := 0;
    signal EventAveraged : signed(15 downto 0) := (others => '0');

    signal EventAverageOut : signed(15 downto 0) := (others => '0');
    signal EventAverageOutUpdated : std_logic := '0';


    signal OutputDataArray : StorageArrayOut := (others => (others => '0'));
    signal OutputSum : signed(16 + LOG_AVG_LEN - 1 downto 0) := (others => '0');

    procedure AverageDataArrayGate(signal RollingArray : inout StorageArrayGate;
    						   signal Sumdata : inout signed(16 + LOG_GATE_LEN - 1 downto 0);
                               signal Counter : inout integer;
                               signal ADCValue : in signed(15 downto 0);
                               signal Averaged : out signed(15 downto 0)) is
    begin
        RollingArray <= ADCValue & RollingArray(0 to 2 ** LOG_GATE_LEN - 2);
        Sumdata <= Sumdata + ADCValue - RollingArray(2 ** LOG_GATE_LEN - 1);
    	Averaged <= Sumdata(16 + LOG_GATE_LEN - 1 downto LOG_GATE_LEN);
        Counter <= Counter + 1;
    end procedure;

    procedure AverageDataArrayOut(signal RollingArray : inout StorageArrayOut;
    						   signal Sumdata : inout signed(16 + LOG_AVG_LEN - 1 downto 0);
                               signal ADCValue : in signed(15 downto 0)) is
    begin
        RollingArray <= ADCValue & RollingArray(0 to 2 ** LOG_AVG_LEN - 2);
        Sumdata <= Sumdata + ADCValue - RollingArray(2 ** LOG_AVG_LEN - 1);
    end procedure;


begin
	process(Clk) is
	begin
        if rising_edge(Clk) then
            if Reset then
                EventAverageOutUpdated <= '0';
                PreviousInputB <= (others => '0');
                State <= WaitForTrigger;
            else
                case State is
                    when WaitForTrigger =>
                        EventAverageOutUpdated <= '0';
                        if (InputB > to_signed(2000, InputB'length)) and (PreviousInputB < to_signed(2000, PreviousInputB'length)) then
                            State <= Triggered;
                            EventCounter <= 0;
                            EventSum <= (others => '0');
                            EventDataArray <= (others => (others => '0'));
                        end if;
                        PreviousInputB <= InputB;

                    when Triggered =>
                        AverageDataArrayGate(EventDataArray, EventSum, EventCounter, InputA, EventAveraged);
                        if EventCounter = 2 ** LOG_GATE_LEN + 1 then
                            EventAverageOut <= EventAveraged;
                            EventAverageOutUpdated <= '1';
                            State <= WaitForTrigger;
                        end if;
                end case;
            end if;
        end if;
    end process;

    process(Clk) is
    begin
        if rising_edge(Clk) then
          if EventAverageOutUpdated then
              AverageDataArrayOut(OutputDataArray, OutputSum, EventAverageOut);
          end if;
        end if;
    end process;

    OutputA <= OutputSum(16 + LOG_AVG_LEN - 1 downto LOG_AVG_LEN);
end architecture;
