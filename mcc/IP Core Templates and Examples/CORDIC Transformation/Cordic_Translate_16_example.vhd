LIBRARY ieee;

ARCHITECTURE Behavioral OF CustomWrapper IS

	SIGNAL m_axis_dout_tvalid : STD_LOGIC;
	SIGNAL tdata_temp : signed(31 DOWNTO 0);

BEGIN
	Cordic : Cordic_Translate_16
	PORT MAP(
		aclk => clk,
		-- input is always valid
		s_axis_cartesian_tvalid => '1',
		-- InputA : imaginary part
		-- InputB : real part
		s_axis_cartesian_tdata => STD_LOGIC_VECTOR(InputA & InputB),
		m_axis_dout_tvalid => m_axis_dout_tvalid,
		m_axis_dout_tdata => tdata_temp
	);

	PROCESS (clk)
	BEGIN
		IF rising_edge(clk) THEN

			IF m_axis_dout_tvalid THEN
				OutputA <= signed(tdata_temp(15 DOWNTO 0)); -- amplitude
				OutputB <= signed(tdata_temp(31 DOWNTO 16));-- phase
			END IF;

		END IF;
	END PROCESS;

END ARCHITECTURE;