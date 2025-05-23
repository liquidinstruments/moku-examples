LIBRARY ieee;

ARCHITECTURE Behavioural OF CustomWrapper IS

	SIGNAL FIR_out_temp : STD_LOGIC_VECTOR(23 DOWNTO 0);

BEGIN

	FIR_Filter : FIR_Filter_7coef
	PORT MAP(
		aclk => clk,
		-- input data is always valid
		s_axis_data_tvalid => '1',
		-- FIR filter ready to accept data
		-- leave it open
		s_axis_data_tready => OPEN,
		s_axis_data_tdata => InputA,

		-- FIR filtered data is available to be transferred 
		m_axis_data_tvalid => OPEN,
		m_axis_data_tdata => FIR_out_temp
	);

	OutputA <= signed(FIR_out_temp(23 DOWNTO 8));

END ARCHITECTURE;