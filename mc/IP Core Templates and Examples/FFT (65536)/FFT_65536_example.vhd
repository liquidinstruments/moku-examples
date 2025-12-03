LIBRARY ieee;

ARCHITECTURE Behavioural OF CustomInstrument IS

	TYPE array_tuser IS ARRAY (0 TO 19) OF signed(15 DOWNTO 0);

	SIGNAL aresetn, aresetn_dly : STD_LOGIC;

	SIGNAL s_axis_data_tdata : STD_LOGIC_VECTOR(31 DOWNTO 0);
	SIGNAL s_axis_data_tready : STD_LOGIC;

	SIGNAL m_axis_data_tlast : STD_LOGIC;
	SIGNAL m_axis_data_tvalid, m_axis_data_tvalid_dly : STD_LOGIC;
	SIGNAL count_out, m_axis_data_tuser : signed(15 DOWNTO 0);
	SIGNAL m_axis_data_tuser_dly : array_tuser;

	SIGNAL fftdata_temp : signed(31 DOWNTO 0);
	SIGNAL real, im : signed(15 DOWNTO 0);

	SIGNAL m_axis_dout_tvalid : STD_LOGIC;
	SIGNAL tdata_temp : signed(31 DOWNTO 0);

BEGIN

	FFT_DUT : FFT_65536
	PORT MAP(
		aclk => clk,
		-- aresetn => (not Reset) and FFT_reset and FFT_reset_dly,
		aresetn => (NOT reset) AND aresetn AND aresetn_dly,
		-- Control FFT direction with LSB of Control(1)
		-- and scale of the FFT output with Control(0)
		s_axis_config_tdata => "0000000" & Control(0)(31 DOWNTO 0) & Control(1)(0),
		-- Config data is always valid
		s_axis_config_tvalid => '1',
		-- Leave config ready signal open
		s_axis_config_tready => OPEN,
		-- Input only has real values
		s_axis_data_tdata => s_axis_data_tdata,
		-- Input data is always valid
		s_axis_data_tvalid => '1',
		-- Data ready logic
		s_axis_data_tready => s_axis_data_tready,
		-- Continuous data stream
		-- don't have last sample
		s_axis_data_tlast => '0',

		-- Transformed data
		m_axis_data_tdata => fftdata_temp,
		-- FFT frequency index
		m_axis_data_tuser => m_axis_data_tuser,
		-- output is valid
		m_axis_data_tvalid => m_axis_data_tvalid,
		-- Slave device is always ready to accept output
		m_axis_data_tready => '1',
		-- last sample of the frame
		m_axis_data_tlast => m_axis_data_tlast,

		-- don't care events
		event_frame_started => OPEN,
		event_tlast_unexpected => OPEN,
		event_tlast_missing => OPEN,
		event_status_channel_halt => OPEN,
		event_data_in_channel_halt => OPEN,
		event_data_out_channel_halt => OPEN
	);

	-- only output data when data is valid
	PROCESS (clk)
	BEGIN
		IF rising_edge(clk) THEN

			IF s_axis_data_tready THEN
				s_axis_data_tdata <= x"0000" & STD_LOGIC_VECTOR(InputA);
			END IF;

			IF m_axis_data_tvalid THEN
				-- real part
				real <= fftdata_temp(15 DOWNTO 0);
				-- imaginary part
				im <= fftdata_temp(31 DOWNTO 16);
			END IF;
			-- delay 20 clk cycles
			m_axis_data_tuser_dly <= m_axis_data_tuser & m_axis_data_tuser_dly(0 TO 18);
		END IF;
	END PROCESS;
	-- reset process
	-- reset fft when the tlast is high
	PROCESS (clk)
	BEGIN
		IF rising_edge(Clk) THEN

			aresetn_dly <= aresetn;
			IF m_axis_data_tlast THEN
				aresetn <= '0';
			ELSE
				aresetn <= '1'; 
			END IF; 
		END IF; 
	END PROCESS; 

	Cordic : Cordic_Translate_16 
	PORT MAP( aclk => clk,
		s_axis_cartesian_tvalid => m_axis_data_tvalid,
		s_axis_cartesian_tdata => STD_LOGIC_VECTOR(im & real),
		m_axis_dout_tvalid => m_axis_dout_tvalid,
		m_axis_dout_tdata => tdata_temp
	);

	PROCESS (clk)
	BEGIN
		IF rising_edge(clk) THEN

			IF m_axis_dout_tvalid THEN
				OutputA <= signed(tdata_temp(15 DOWNTO 0));
				OutputB <= signed(tdata_temp(31 DOWNTO 16));
				OutputC <= signed(m_axis_data_tuser_dly(19));
			END IF;

		END IF;
	END PROCESS;

END ARCHITECTURE;