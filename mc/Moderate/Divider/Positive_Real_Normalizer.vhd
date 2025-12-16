-----------------------------------------------------
--
-- Positive Real Normalizer
-- computing the normalization of the real input signal 'u'
-- The fixed-point normalization uses a binary search
-- method, finding the normalized output 'z' and 
-- scaling factor 'e'
--
-----------------------------------------------------


LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;
USE work.DSP_pkg.ALL;

ENTITY Positive_Real_Normalizer IS
  PORT( clk                               :   IN    std_logic;
        reset                             :   IN    std_logic;
        u                                 :   IN    std_logic_vector(15 DOWNTO 0);  -- int16
        validIn                           :   IN    std_logic;
        x                                 :   OUT   std_logic_vector(15 DOWNTO 0);  -- sfix16_En14
        e                                 :   OUT   std_logic_vector(31 DOWNTO 0);  -- int32
        isNegative                        :   OUT   std_logic;
        validOut                          :   OUT   std_logic
        );
END Positive_Real_Normalizer;


ARCHITECTURE rtl OF Positive_Real_Normalizer IS

  -- Signals
  SIGNAL u_signed                         : signed(15 DOWNTO 0);  -- int16
  SIGNAL x_tmp                            : signed(15 DOWNTO 0);  -- sfix16_En14
  SIGNAL e_tmp                            : signed(31 DOWNTO 0);  -- int32
  SIGNAL validReg                         : std_logic_vector(4 DOWNTO 0);  -- boolean [5]
  SIGNAL tReg                             : vector_of_unsigned4(0 TO 4);  -- ufix4 [5]
  SIGNAL xReg                             : vector_of_unsigned16(0 TO 4);  -- ufix16 [5]
  SIGNAL isNegativeReg                    : std_logic_vector(4 DOWNTO 0);  -- boolean [5]
  SIGNAL validReg_next                    : std_logic_vector(4 DOWNTO 0);  -- boolean [5]
  SIGNAL tReg_next                        : vector_of_unsigned4(0 TO 4);  -- ufix4 [5]
  SIGNAL xReg_next                        : vector_of_unsigned16(0 TO 4);  -- ufix16 [5]
  SIGNAL isNegativeReg_next               : std_logic_vector(4 DOWNTO 0);  -- boolean [5]

BEGIN
  u_signed <= signed(u);

  embreciprocals_c3_positiveRealNormalizer_process : PROCESS (clk, reset)
  BEGIN
    IF reset = '1' THEN
      validReg <= (OTHERS => '0');
      tReg <= (OTHERS => to_unsigned(16#0#, 4));
      xReg <= (OTHERS => to_unsigned(16#0000#, 16));
      isNegativeReg <= (OTHERS => '0');
    ELSIF clk'EVENT AND clk = '1' THEN
      validReg <= validReg_next;
      tReg <= tReg_next;
      xReg <= xReg_next;
      isNegativeReg <= isNegativeReg_next;
    END IF;
  END PROCESS embreciprocals_c3_positiveRealNormalizer_process;

  embreciprocals_c3_positiveRealNormalizer_output : PROCESS (isNegativeReg, tReg, u_signed, validIn, validReg, xReg)
    VARIABLE u1 : signed(15 DOWNTO 0);
    VARIABLE isNegative1 : std_logic;
    VARIABLE a : unsigned(15 DOWNTO 0);
    VARIABLE c : unsigned(15 DOWNTO 0);
    VARIABLE xfi_stripped : unsigned(15 DOWNTO 0);
    VARIABLE yfi_trivial_scaling : signed(15 DOWNTO 0);
    VARIABLE a_0 : unsigned(15 DOWNTO 0);
    VARIABLE c_0 : unsigned(15 DOWNTO 0);
    VARIABLE a_1 : unsigned(15 DOWNTO 0);
    VARIABLE c_1 : unsigned(15 DOWNTO 0);
    VARIABLE a_2 : unsigned(15 DOWNTO 0);
    VARIABLE c_2 : unsigned(15 DOWNTO 0);
    VARIABLE a_3 : unsigned(15 DOWNTO 0);
    VARIABLE c_3 : unsigned(15 DOWNTO 0);
    VARIABLE t_0 : signed(16 DOWNTO 0);
    VARIABLE sub_cast : signed(31 DOWNTO 0);
  BEGIN
    c_3 := to_unsigned(16#0000#, 16);
    c_2 := to_unsigned(16#0000#, 16);
    c_1 := to_unsigned(16#0000#, 16);
    c_0 := to_unsigned(16#0000#, 16);
    a_3 := to_unsigned(16#0000#, 16);
    a_2 := to_unsigned(16#0000#, 16);
    a_1 := to_unsigned(16#0000#, 16);
    a_0 := to_unsigned(16#0000#, 16);
    t_0 := to_signed(16#00000#, 17);

    --realNormalizer Normalize real values.
    -- Given real scalar u ~= 0, this block produces x such that
    --    1 <= x < 2,
    -- e such that
    --    x = (2^e)*|u|,
    -- and isNegative = u<0.
    --
    --   When u = 0 and u is fixed-point or scaled-double, then x = 0 and
    --   e = (2^nextpow2(x.WordLength)) - x.WordLength - x.FractionLength.
    --
    --   When u = 0 and u is floating-point, then x = 0 and e = 1.
    --   Copyright 2019 The MathWorks, Inc.
    -- This function only works on scalars.
    -- Only operate on the real part
    -- Normalize in unsigned type.

    u1 := u_signed;
    IF u_signed < to_signed(16#0000#, 16) THEN 
      isNegative1 := '1';
    ELSE 
      isNegative1 := '0';
    END IF;
    IF isNegative1 = '1' THEN 
      t_0 :=  - (resize(u_signed, 17));
      IF (t_0(16) = '0') AND (t_0(15) /= '0') THEN 
        u1 := X"7FFF";
      ELSIF (t_0(16) = '1') AND (t_0(15) /= '1') THEN 
        u1 := X"8000";
      ELSE 
        u1 := t_0(15 DOWNTO 0);
      END IF;
    END IF;
    -- Normalize fixed-point values
    -- Normalize fixed-point values.
    -- For fixed-point types, the normalization uses a binary search of
    -- length log2 of the word length of the input.
    validReg_next(4) <= validReg(3);
    validReg_next(3) <= validReg(2);
    validReg_next(2) <= validReg(1);
    validReg_next(1) <= validReg(0);
    validReg_next(0) <= validIn;
    IF (xReg(3) AND to_unsigned(16#8000#, 16)) = to_unsigned(16#0000#, 16) THEN 
      tReg_next(4) <= tReg(3) OR to_unsigned(16#1#, 4);
      a_0 := xReg(3);
      c_0 := a_0 sll 1;
      xReg_next(4) <= c_0;
    ELSE 
      tReg_next(4) <= tReg(3);
      xReg_next(4) <= xReg(3);
    END IF;
    IF (xReg(2) AND to_unsigned(16#C000#, 16)) = to_unsigned(16#0000#, 16) THEN 
      tReg_next(3) <= tReg(2) OR to_unsigned(16#2#, 4);
      a_1 := xReg(2);
      c_1 := a_1 sll 2;
      xReg_next(3) <= c_1;
    ELSE 
      tReg_next(3) <= tReg(2);
      xReg_next(3) <= xReg(2);
    END IF;
    IF (xReg(1) AND to_unsigned(16#F000#, 16)) = to_unsigned(16#0000#, 16) THEN 
      tReg_next(2) <= tReg(1) OR to_unsigned(16#4#, 4);
      a_2 := xReg(1);
      c_2 := a_2 sll 4;
      xReg_next(2) <= c_2;
    ELSE 
      tReg_next(2) <= tReg(1);
      xReg_next(2) <= xReg(1);
    END IF;
    IF (xReg(0) AND to_unsigned(16#FF00#, 16)) = to_unsigned(16#0000#, 16) THEN 
      tReg_next(1) <= tReg(0) OR to_unsigned(16#8#, 4);
      a_3 := xReg(0);
      c_3 := a_3 sll 8;
      xReg_next(1) <= c_3;
    ELSE 
      tReg_next(1) <= tReg(0);
      xReg_next(1) <= xReg(0);
    END IF;
    tReg_next(0) <= to_unsigned(16#0#, 4);
    xReg_next(0) <= unsigned(u1);

    --  % Persistent
    -- Assign outputs from states
    -- Update isNegative states

    isNegativeReg_next(4) <= isNegativeReg(3);
    isNegativeReg_next(3) <= isNegativeReg(2);
    isNegativeReg_next(2) <= isNegativeReg(1);
    isNegativeReg_next(1) <= isNegativeReg(0);
    isNegativeReg_next(0) <= isNegative1;
    -- Cast the output to signed if the input was signed.
    a := xReg(4);
    c := SHIFT_RIGHT(a, 1);
    xfi_stripped := c;
    yfi_trivial_scaling := signed(xfi_stripped);

    -- Convert the normalized shift value based on the data type of the
    -- input U so that the output of the normalizer X is in real-world
    -- scale and X = (2^N)*U.
    
    x_tmp <= yfi_trivial_scaling;
    sub_cast := signed(resize(tReg(4), 32));
    e_tmp <= sub_cast - 15;
    isNegative <= isNegativeReg(4);
    validOut <= validReg(4);
  END PROCESS embreciprocals_c3_positiveRealNormalizer_output;


  x <= std_logic_vector(x_tmp);

  e <= std_logic_vector(e_tmp);

END rtl;

