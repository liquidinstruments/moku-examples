LIBRARY ieee;

ARCHITECTURE Behavioural OF CustomInstrument IS

	SIGNAL sclr_triangular : STD_LOGIC;
	SIGNAL sclr_sawtooth : STD_LOGIC;

	SIGNAL up_triangular : STD_LOGIC;
	SIGNAL up_sawtooth : STD_LOGIC;

	SIGNAL q_triangular : STD_LOGIC_VECTOR(31 DOWNTO 0);
	SIGNAL q_sawtooth : STD_LOGIC_VECTOR(31 DOWNTO 0);

BEGIN

	-- Triangular wave
	Triangular : Counter_32
	PORT MAP(
		clk => clk,
		-- synchronous clear
		sclr => sclr_triangular,
		up => up_triangular,
		q => q_triangular
	);
	OutputA <= signed(q_triangular(15 DOWNTO 0));

	-- Triangular counter configuration
	PROCESS (clk) IS
	BEGIN
		IF rising_edge(clk) THEN
			-- reset
			IF Control(0)(0) = '1' THEN
				sclr_triangular <= '1';
				up_triangular <= '1';
			ELSE
				-- don't clear 
				sclr_triangular <= '0';
				-- Control1: counter limit
				IF q_triangular = Control(1) THEN
					-- count down
					up_triangular <= '0';
				ELSIF q_triangular = x"00000000" THEN
					-- count up
					up_triangular <= '1';
				ELSE
					-- hold
					up_triangular <= up_triangular; 
				END IF; 
			END IF; 
		END IF; 
	END PROCESS; 
	
	Sawtooth : Counter_32 
	PORT MAP( 
		clk => clk,
		sclr => sclr_sawtooth, -- synchronous clear
		up => up_sawtooth,
		q => q_sawtooth
	);
	OutputB <= signed(q_sawtooth(15 DOWNTO 0));

	-- Sawtooth counter configuration
	PROCESS (clk) IS
	BEGIN
		IF rising_edge(clk) THEN
			-- reset
			IF Control(0)(0) = '1' THEN
				sclr_sawtooth <= '1';
				up_sawtooth <= '1';
			ELSE
				-- always count up
				up_sawtooth <= '1';
				-- Control2 : counter limit
				IF q_sawtooth = Control(2) THEN
					-- clear
					sclr_sawtooth <= '1';
				ELSE
					-- continue counting
					sclr_sawtooth <= '0';
				END IF;
			END IF;
		END IF;
	END PROCESS;

END ARCHITECTURE;