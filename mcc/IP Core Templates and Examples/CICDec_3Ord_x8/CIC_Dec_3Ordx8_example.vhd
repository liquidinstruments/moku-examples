LIBRARY ieee;

ARCHITECTURE Behavioral OF CustomWrapper IS

	SIGNAL s_axis_data_tready : STD_LOGIC;
	SIGNAL s_axis_data_tdata : STD_LOGIC_VECTOR(15 DOWNTO 0);

	SIGNAL m_axis_data_tvalid : STD_LOGIC;
	SIGNAL m_axis_data_tdata : STD_LOGIC_VECTOR(31 DOWNTO 0);

BEGIN

	Decimator : CIC_Dec_3Ordx8
	PORT MAP(
		aclk => clk,
		s_axis_data_tdata => s_axis_data_tdata,
		s_axis_data_tvalid => '1', -- always output
		s_axis_data_tready => s_axis_data_tready,
		m_axis_data_tdata => m_axis_data_tdata,
		m_axis_data_tvalid => m_axis_data_tvalid
	);

	PROCESS (clk)
	BEGIN
		IF rising_edge(clk) THEN

			-- update input data when CIC is ready
			IF s_axis_data_tready THEN
				s_axis_data_tdata <= STD_LOGIC_VECTOR(InputA);
			ELSE
				s_axis_data_tdata <= (OTHERS => '0');
			END IF;

			-- update Output only when data is valid
			IF m_axis_data_tvalid THEN
				-- Scale data correctly
				OutputA <= signed(m_axis_data_tdata(24 DOWNTO 9));
			END IF;

		END IF;
	END PROCESS;

END ARCHITECTURE;