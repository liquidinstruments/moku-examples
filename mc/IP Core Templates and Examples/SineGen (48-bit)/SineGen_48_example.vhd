LIBRARY ieee;

ARCHITECTURE Behavioural OF CustomInstrument IS

	SIGNAL m_axis_data_tvalid : STD_LOGIC;
	SIGNAL m_axis_phase_tvalid : STD_LOGIC;
	SIGNAL sine_temp : STD_LOGIC_VECTOR(31 DOWNTO 0);
	SIGNAL m_axis_phase_tdata : STD_LOGIC_VECTOR(47 DOWNTO 0);

BEGIN

	SineCosineGen : SineGen_48
	PORT MAP(
		aclk => clk,
		-- Use the 0th bit of Control(0) to reset this module
		aresetn => NOT Control(0)(0),
		-- input signal is always available
		s_axis_config_tvalid => '1',
		-- 48-bit frequency step
		s_axis_config_tdata => Control(2)(15 DOWNTO 0) & Control(1),

		m_axis_data_tvalid => m_axis_data_tvalid,
		m_axis_data_tdata => sine_temp,

		m_axis_phase_tvalid => m_axis_phase_tvalid,
		m_axis_phase_tdata => m_axis_phase_tdata -- 48-bit phase counter output
	);

	-- only output data when data is valid
	PROCESS (clk)
	BEGIN
		IF rising_edge(clk) THEN

			IF m_axis_data_tvalid THEN
				-- 32-bit, the most significant 16 bits are sine
				-- and the least significant 16 bits are cosine
				OutputA <= signed(sine_temp(15 DOWNTO 0));
				OutputB <= signed(sine_temp(31 DOWNTO 16));
			END IF;

			IF m_axis_phase_tvalid THEN
				OutputC <= signed(m_axis_phase_tdata(47 DOWNTO 32));
			END IF;

		END IF;
	END PROCESS;

END ARCHITECTURE;