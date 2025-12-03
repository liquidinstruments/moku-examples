LIBRARY ieee;

ARCHITECTURE Behavioral OF CustomInstrument IS

	SIGNAL s_temp : STD_LOGIC_VECTOR(15 DOWNTO 0);

BEGIN

	AddSubtract : AddSubtract_16
	PORT MAP(
		A => STD_LOGIC_VECTOR(InputA),
		B => STD_LOGIC_VECTOR(InputB),
		clk => clk,
		-- use Control(0)'s 0th bit to control the operation
		add => Control(0)(0),
		-- constant high clock enable
		ce => '1',
		S => s_temp
	);

	OutputA <= signed(s_temp);

END ARCHITECTURE;