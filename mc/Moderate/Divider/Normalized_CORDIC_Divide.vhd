-----------------------------------------------------
--
-- Normalized CORDIC Divide
-- impliments hardware module for performing fixed-point
-- division using CORDIC (Coordinate Rotation Digital Computer)
-- algorithm. Iteratively performing CORDIC vector
-- rotations to find output quotient 'y' and remainder 't' 
-- of division operation
--
-----------------------------------------------------


LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;
USE work.DSP_pkg.ALL;

ENTITY Normalized_CORDIC_Divide IS
  PORT( clk                               :   IN    std_logic;
        reset                             :   IN    std_logic;
        num                               :   IN    std_logic_vector(15 DOWNTO 0);  -- sfix16_En14
        den                               :   IN    std_logic_vector(15 DOWNTO 0);  -- sfix16_En14
        tNum                              :   IN    std_logic_vector(32 DOWNTO 0);  -- sfix33
        tDen                              :   IN    std_logic_vector(31 DOWNTO 0);  -- int32
        isNumNegative                     :   IN    std_logic;
        isDenNegative                     :   IN    std_logic;
        validIn                           :   IN    std_logic;
        y                                 :   OUT   std_logic_vector(15 DOWNTO 0);  -- sfix16_En14
        t                                 :   OUT   std_logic_vector(33 DOWNTO 0);  -- sfix34
        isDenZeroOut                      :   OUT   std_logic;
        validOut                          :   OUT   std_logic
        );
END Normalized_CORDIC_Divide;


ARCHITECTURE rtl OF Normalized_CORDIC_Divide IS

  -- Signals
  SIGNAL num_signed                       : signed(15 DOWNTO 0);  -- sfix16_En14
  SIGNAL den_signed                       : signed(15 DOWNTO 0);  -- sfix16_En14
  SIGNAL tNum_signed                      : signed(32 DOWNTO 0);  -- sfix33
  SIGNAL tDen_signed                      : signed(31 DOWNTO 0);  -- int32
  SIGNAL y_tmp                            : signed(15 DOWNTO 0);  -- sfix16_En14
  SIGNAL t_tmp                            : signed(33 DOWNTO 0);  -- sfix34
  SIGNAL xReg                             : vector_of_signed16(0 TO 16);  -- sfix16 [17]
  SIGNAL yReg                             : vector_of_signed16(0 TO 16);  -- sfix16 [17]
  SIGNAL zReg                             : vector_of_signed16(0 TO 16);  -- sfix16 [17]
  SIGNAL validReg                         : std_logic_vector(16 DOWNTO 0);  -- boolean [17]
  SIGNAL tReg                             : vector_of_signed34(0 TO 16);  -- sfix34 [17]
  SIGNAL isNegativeReg                    : std_logic_vector(16 DOWNTO 0);  -- boolean [17]
  SIGNAL isNumZeroReg                     : std_logic_vector(16 DOWNTO 0);  -- boolean [17]
  SIGNAL isDenZeroReg                     : std_logic_vector(16 DOWNTO 0);  -- boolean [17]
  SIGNAL xReg_next                        : vector_of_signed16(0 TO 16);  -- sfix16_En14 [17]
  SIGNAL yReg_next                        : vector_of_signed16(0 TO 16);  -- sfix16_En14 [17]
  SIGNAL zReg_next                        : vector_of_signed16(0 TO 16);  -- sfix16_En14 [17]
  SIGNAL validReg_next                    : std_logic_vector(16 DOWNTO 0);  -- boolean [17]
  SIGNAL tReg_next                        : vector_of_signed34(0 TO 16);  -- sfix34 [17]
  SIGNAL isNegativeReg_next               : std_logic_vector(16 DOWNTO 0);  -- boolean [17]
  SIGNAL isNumZeroReg_next                : std_logic_vector(16 DOWNTO 0);  -- boolean [17]
  SIGNAL isDenZeroReg_next                : std_logic_vector(16 DOWNTO 0);  -- boolean [17]

BEGIN
  num_signed <= signed(num);

  den_signed <= signed(den);

  tNum_signed <= signed(tNum);

  tDen_signed <= signed(tDen);

  embreciprocals_c21_normalizedCORDICDivide_process : PROCESS (clk, reset)
  BEGIN
    IF reset = '1' THEN
      xReg <= (OTHERS => to_signed(16#0000#, 16));
      yReg <= (OTHERS => to_signed(16#0000#, 16));
      zReg <= (OTHERS => to_signed(16#0000#, 16));
      validReg <= (OTHERS => '0');
      tReg <= (OTHERS => to_signed(0, 34));
      isNegativeReg <= (OTHERS => '0');
      isDenZeroReg <= (OTHERS => '0');
      isNumZeroReg <= (OTHERS => '0');
    ELSIF clk'EVENT AND clk = '1' THEN
      xReg <= xReg_next;
      yReg <= yReg_next;
      zReg <= zReg_next;
      validReg <= validReg_next;
      tReg <= tReg_next;
      isNegativeReg <= isNegativeReg_next;
      isNumZeroReg <= isNumZeroReg_next;
      isDenZeroReg <= isDenZeroReg_next;
    END IF;
  END PROCESS embreciprocals_c21_normalizedCORDICDivide_process;

  embreciprocals_c21_normalizedCORDICDivide_output : PROCESS (den_signed, isDenNegative, isDenZeroReg, isNegativeReg, isNumNegative,
       isNumZeroReg, num_signed, tDen_signed, tNum_signed, tReg, validIn,
       validReg, xReg, yReg, zReg)
    VARIABLE isDenZero : std_logic;
    VARIABLE isNumZero : std_logic;
    VARIABLE sub_temp : signed(15 DOWNTO 0);
    VARIABLE t_0 : signed(15 DOWNTO 0);
    VARIABLE a : signed(15 DOWNTO 0);
    VARIABLE c : signed(15 DOWNTO 0);
    VARIABLE a_0 : signed(15 DOWNTO 0);
    VARIABLE c_0 : signed(15 DOWNTO 0);
    VARIABLE a_1 : signed(15 DOWNTO 0);
    VARIABLE c_1 : signed(15 DOWNTO 0);
    VARIABLE a_2 : signed(15 DOWNTO 0);
    VARIABLE c_2 : signed(15 DOWNTO 0);
    VARIABLE a_3 : signed(15 DOWNTO 0);
    VARIABLE c_3 : signed(15 DOWNTO 0);
    VARIABLE a_4 : signed(15 DOWNTO 0);
    VARIABLE c_4 : signed(15 DOWNTO 0);
    VARIABLE a_5 : signed(15 DOWNTO 0);
    VARIABLE c_5 : signed(15 DOWNTO 0);
    VARIABLE a_6 : signed(15 DOWNTO 0);
    VARIABLE c_6 : signed(15 DOWNTO 0);
    VARIABLE a_7 : signed(15 DOWNTO 0);
    VARIABLE c_7 : signed(15 DOWNTO 0);
    VARIABLE a_8 : signed(15 DOWNTO 0);
    VARIABLE c_8 : signed(15 DOWNTO 0);
    VARIABLE a_9 : signed(15 DOWNTO 0);
    VARIABLE c_9 : signed(15 DOWNTO 0);
    VARIABLE a_10 : signed(15 DOWNTO 0);
    VARIABLE c_10 : signed(15 DOWNTO 0);
    VARIABLE a_11 : signed(15 DOWNTO 0);
    VARIABLE c_11 : signed(15 DOWNTO 0);
    VARIABLE a_12 : signed(15 DOWNTO 0);
    VARIABLE c_12 : signed(15 DOWNTO 0);
    VARIABLE a_13 : signed(15 DOWNTO 0);
    VARIABLE c_13 : signed(15 DOWNTO 0);
    VARIABLE a_14 : signed(15 DOWNTO 0);
    VARIABLE sub_temp_0 : signed(15 DOWNTO 0);
    VARIABLE add_temp : signed(15 DOWNTO 0);
    VARIABLE sub_temp_1 : signed(15 DOWNTO 0);
    VARIABLE add_temp_0 : signed(15 DOWNTO 0);
    VARIABLE sub_temp_2 : signed(15 DOWNTO 0);
    VARIABLE add_temp_1 : signed(15 DOWNTO 0);
    VARIABLE sub_temp_3 : signed(15 DOWNTO 0);
    VARIABLE add_temp_2 : signed(15 DOWNTO 0);
    VARIABLE sub_temp_4 : signed(15 DOWNTO 0);
    VARIABLE add_temp_3 : signed(15 DOWNTO 0);
    VARIABLE sub_temp_5 : signed(15 DOWNTO 0);
    VARIABLE add_temp_4 : signed(15 DOWNTO 0);
    VARIABLE sub_temp_6 : signed(15 DOWNTO 0);
    VARIABLE add_temp_5 : signed(15 DOWNTO 0);
    VARIABLE sub_temp_7 : signed(15 DOWNTO 0);
    VARIABLE add_temp_6 : signed(15 DOWNTO 0);
    VARIABLE sub_temp_8 : signed(15 DOWNTO 0);
    VARIABLE add_temp_7 : signed(15 DOWNTO 0);
    VARIABLE sub_temp_9 : signed(15 DOWNTO 0);
    VARIABLE add_temp_8 : signed(15 DOWNTO 0);
    VARIABLE sub_temp_10 : signed(15 DOWNTO 0);
    VARIABLE add_temp_9 : signed(15 DOWNTO 0);
    VARIABLE sub_temp_11 : signed(15 DOWNTO 0);
    VARIABLE add_temp_10 : signed(15 DOWNTO 0);
    VARIABLE sub_temp_12 : signed(15 DOWNTO 0);
    VARIABLE add_temp_11 : signed(15 DOWNTO 0);
    VARIABLE sub_temp_13 : signed(15 DOWNTO 0);
    VARIABLE add_temp_12 : signed(15 DOWNTO 0);
    VARIABLE sub_temp_14 : signed(15 DOWNTO 0);
    VARIABLE add_temp_13 : signed(15 DOWNTO 0);
    VARIABLE cast : signed(16 DOWNTO 0);
    VARIABLE cast_0 : signed(16 DOWNTO 0);
  BEGIN
    cast_0 := to_signed(16#00000#, 17);
    cast := to_signed(16#00000#, 17);
    --   Copyright 2020 The MathWorks, Inc.
    IF den_signed = to_signed(16#0000#, 16) THEN 
      isDenZero := '1';
    ELSE 
      isDenZero := '0';
    END IF;
    IF num_signed = to_signed(16#0000#, 16) THEN 
      isNumZero := '1';
    ELSE 
      isNumZero := '0';
    END IF;
    -- CORDIC divide for fixed-point and scaled-double types
    -- Register variables for the CORDIC Divide Kernel
    -- CORDIC Divide Kernel
    validReg_next(16) <= validReg(15);
    tReg_next(16) <= tReg(15);
    validReg_next(15) <= validReg(14);
    tReg_next(15) <= tReg(14);
    validReg_next(14) <= validReg(13);
    tReg_next(14) <= tReg(13);
    validReg_next(13) <= validReg(12);
    tReg_next(13) <= tReg(12);
    validReg_next(12) <= validReg(11);
    tReg_next(12) <= tReg(11);
    validReg_next(11) <= validReg(10);
    tReg_next(11) <= tReg(10);
    validReg_next(10) <= validReg(9);
    tReg_next(10) <= tReg(9);
    validReg_next(9) <= validReg(8);
    tReg_next(9) <= tReg(8);
    validReg_next(8) <= validReg(7);
    tReg_next(8) <= tReg(7);
    validReg_next(7) <= validReg(6);
    tReg_next(7) <= tReg(6);
    validReg_next(6) <= validReg(5);
    tReg_next(6) <= tReg(5);
    validReg_next(5) <= validReg(4);
    tReg_next(5) <= tReg(4);
    validReg_next(4) <= validReg(3);
    tReg_next(4) <= tReg(3);
    validReg_next(3) <= validReg(2);
    tReg_next(3) <= tReg(2);
    validReg_next(2) <= validReg(1);
    tReg_next(2) <= tReg(1);
    validReg_next(1) <= validReg(0);
    tReg_next(1) <= tReg(0);
    validReg_next(0) <= validIn;
    tReg_next(0) <= resize(tDen_signed, 34) - resize(tNum_signed, 34);
    a := xReg(15);
    c := SHIFT_RIGHT(a, 15);
    IF yReg(15) < to_signed(16#0000#, 16) THEN 
      sub_temp := yReg(15) + c;
      t_0 := zReg(15);
    ELSE 
      sub_temp := yReg(15) - c;
      t_0 := zReg(15);
    END IF;
    yReg_next(16) <= sub_temp;
    zReg_next(16) <= t_0;
    xReg_next(16) <= xReg(15);
    a_0 := xReg(14);
    c_0 := SHIFT_RIGHT(a_0, 14);
    IF yReg(14) < to_signed(16#0000#, 16) THEN 
      sub_temp_0 := yReg(14) + c_0;
      add_temp := zReg(14) - to_signed(16#0001#, 16);
    ELSE 
      sub_temp_0 := yReg(14) - c_0;
      add_temp := zReg(14) + to_signed(16#0001#, 16);
    END IF;
    yReg_next(15) <= sub_temp_0;
    zReg_next(15) <= add_temp;
    xReg_next(15) <= xReg(14);
    a_1 := xReg(13);
    c_1 := SHIFT_RIGHT(a_1, 13);
    IF yReg(13) < to_signed(16#0000#, 16) THEN 
      sub_temp_1 := yReg(13) + c_1;
      add_temp_0 := zReg(13) - to_signed(16#0002#, 16);
    ELSE 
      sub_temp_1 := yReg(13) - c_1;
      add_temp_0 := zReg(13) + to_signed(16#0002#, 16);
    END IF;
    yReg_next(14) <= sub_temp_1;
    zReg_next(14) <= add_temp_0;
    xReg_next(14) <= xReg(13);
    a_2 := xReg(12);
    c_2 := SHIFT_RIGHT(a_2, 12);
    IF yReg(12) < to_signed(16#0000#, 16) THEN 
      sub_temp_2 := yReg(12) + c_2;
      add_temp_1 := zReg(12) - to_signed(16#0004#, 16);
    ELSE 
      sub_temp_2 := yReg(12) - c_2;
      add_temp_1 := zReg(12) + to_signed(16#0004#, 16);
    END IF;
    yReg_next(13) <= sub_temp_2;
    zReg_next(13) <= add_temp_1;
    xReg_next(13) <= xReg(12);
    a_3 := xReg(11);
    c_3 := SHIFT_RIGHT(a_3, 11);
    IF yReg(11) < to_signed(16#0000#, 16) THEN 
      sub_temp_3 := yReg(11) + c_3;
      add_temp_2 := zReg(11) - to_signed(16#0008#, 16);
    ELSE 
      sub_temp_3 := yReg(11) - c_3;
      add_temp_2 := zReg(11) + to_signed(16#0008#, 16);
    END IF;
    yReg_next(12) <= sub_temp_3;
    zReg_next(12) <= add_temp_2;
    xReg_next(12) <= xReg(11);
    a_4 := xReg(10);
    c_4 := SHIFT_RIGHT(a_4, 10);
    IF yReg(10) < to_signed(16#0000#, 16) THEN 
      sub_temp_4 := yReg(10) + c_4;
      add_temp_3 := zReg(10) - to_signed(16#0010#, 16);
    ELSE 
      sub_temp_4 := yReg(10) - c_4;
      add_temp_3 := zReg(10) + to_signed(16#0010#, 16);
    END IF;
    yReg_next(11) <= sub_temp_4;
    zReg_next(11) <= add_temp_3;
    xReg_next(11) <= xReg(10);
    a_5 := xReg(9);
    c_5 := SHIFT_RIGHT(a_5, 9);
    IF yReg(9) < to_signed(16#0000#, 16) THEN 
      sub_temp_5 := yReg(9) + c_5;
      add_temp_4 := zReg(9) - to_signed(16#0020#, 16);
    ELSE 
      sub_temp_5 := yReg(9) - c_5;
      add_temp_4 := zReg(9) + to_signed(16#0020#, 16);
    END IF;
    yReg_next(10) <= sub_temp_5;
    zReg_next(10) <= add_temp_4;
    xReg_next(10) <= xReg(9);
    a_6 := xReg(8);
    c_6 := SHIFT_RIGHT(a_6, 8);
    IF yReg(8) < to_signed(16#0000#, 16) THEN 
      sub_temp_6 := yReg(8) + c_6;
      add_temp_5 := zReg(8) - to_signed(16#0040#, 16);
    ELSE 
      sub_temp_6 := yReg(8) - c_6;
      add_temp_5 := zReg(8) + to_signed(16#0040#, 16);
    END IF;
    yReg_next(9) <= sub_temp_6;
    zReg_next(9) <= add_temp_5;
    xReg_next(9) <= xReg(8);
    a_7 := xReg(7);
    c_7 := SHIFT_RIGHT(a_7, 7);
    IF yReg(7) < to_signed(16#0000#, 16) THEN 
      sub_temp_7 := yReg(7) + c_7;
      add_temp_6 := zReg(7) - to_signed(16#0080#, 16);
    ELSE 
      sub_temp_7 := yReg(7) - c_7;
      add_temp_6 := zReg(7) + to_signed(16#0080#, 16);
    END IF;
    yReg_next(8) <= sub_temp_7;
    zReg_next(8) <= add_temp_6;
    xReg_next(8) <= xReg(7);
    a_8 := xReg(6);
    c_8 := SHIFT_RIGHT(a_8, 6);
    IF yReg(6) < to_signed(16#0000#, 16) THEN 
      sub_temp_8 := yReg(6) + c_8;
      add_temp_7 := zReg(6) - to_signed(16#0100#, 16);
    ELSE 
      sub_temp_8 := yReg(6) - c_8;
      add_temp_7 := zReg(6) + to_signed(16#0100#, 16);
    END IF;
    yReg_next(7) <= sub_temp_8;
    zReg_next(7) <= add_temp_7;
    xReg_next(7) <= xReg(6);
    a_9 := xReg(5);
    c_9 := SHIFT_RIGHT(a_9, 5);
    IF yReg(5) < to_signed(16#0000#, 16) THEN 
      sub_temp_9 := yReg(5) + c_9;
      add_temp_8 := zReg(5) - to_signed(16#0200#, 16);
    ELSE 
      sub_temp_9 := yReg(5) - c_9;
      add_temp_8 := zReg(5) + to_signed(16#0200#, 16);
    END IF;
    yReg_next(6) <= sub_temp_9;
    zReg_next(6) <= add_temp_8;
    xReg_next(6) <= xReg(5);
    a_10 := xReg(4);
    c_10 := SHIFT_RIGHT(a_10, 4);
    IF yReg(4) < to_signed(16#0000#, 16) THEN 
      sub_temp_10 := yReg(4) + c_10;
      add_temp_9 := zReg(4) - to_signed(16#0400#, 16);
    ELSE 
      sub_temp_10 := yReg(4) - c_10;
      add_temp_9 := zReg(4) + to_signed(16#0400#, 16);
    END IF;
    yReg_next(5) <= sub_temp_10;
    zReg_next(5) <= add_temp_9;
    xReg_next(5) <= xReg(4);
    a_11 := xReg(3);
    c_11 := SHIFT_RIGHT(a_11, 3);
    IF yReg(3) < to_signed(16#0000#, 16) THEN 
      sub_temp_11 := yReg(3) + c_11;
      add_temp_10 := zReg(3) - to_signed(16#0800#, 16);
    ELSE 
      sub_temp_11 := yReg(3) - c_11;
      add_temp_10 := zReg(3) + to_signed(16#0800#, 16);
    END IF;
    yReg_next(4) <= sub_temp_11;
    zReg_next(4) <= add_temp_10;
    xReg_next(4) <= xReg(3);
    a_12 := xReg(2);
    c_12 := SHIFT_RIGHT(a_12, 2);
    IF yReg(2) < to_signed(16#0000#, 16) THEN 
      sub_temp_12 := yReg(2) + c_12;
      add_temp_11 := zReg(2) - to_signed(16#1000#, 16);
    ELSE 
      sub_temp_12 := yReg(2) - c_12;
      add_temp_11 := zReg(2) + to_signed(16#1000#, 16);
    END IF;
    yReg_next(3) <= sub_temp_12;
    zReg_next(3) <= add_temp_11;
    xReg_next(3) <= xReg(2);
    a_13 := xReg(1);
    c_13 := SHIFT_RIGHT(a_13, 1);
    IF yReg(1) < to_signed(16#0000#, 16) THEN 
      sub_temp_13 := yReg(1) + c_13;
      add_temp_12 := zReg(1) - to_signed(16#2000#, 16);
    ELSE 
      sub_temp_13 := yReg(1) - c_13;
      add_temp_12 := zReg(1) + to_signed(16#2000#, 16);
    END IF;
    yReg_next(2) <= sub_temp_13;
    zReg_next(2) <= add_temp_12;
    xReg_next(2) <= xReg(1);
    a_14 := xReg(0);
    IF yReg(0) < to_signed(16#0000#, 16) THEN 
      sub_temp_14 := yReg(0) + a_14;
      add_temp_13 := zReg(0) - to_signed(16#4000#, 16);
    ELSE 
      sub_temp_14 := yReg(0) - a_14;
      add_temp_13 := zReg(0) + to_signed(16#4000#, 16);
    END IF;
    yReg_next(1) <= sub_temp_14;
    zReg_next(1) <= add_temp_13;
    xReg_next(1) <= xReg(0);
    xReg_next(0) <= den_signed;
    yReg_next(0) <= num_signed;
    zReg_next(0) <= to_signed(16#0000#, 16);
    -- % Persistent variables
    -- Assign outputs from states
    -- Update isNegative states
    isNegativeReg_next(16) <= isNegativeReg(15);
    isNumZeroReg_next(16) <= isNumZeroReg(15);
    isDenZeroReg_next(16) <= isDenZeroReg(15);
    isNegativeReg_next(15) <= isNegativeReg(14);
    isNumZeroReg_next(15) <= isNumZeroReg(14);
    isDenZeroReg_next(15) <= isDenZeroReg(14);
    isNegativeReg_next(14) <= isNegativeReg(13);
    isNumZeroReg_next(14) <= isNumZeroReg(13);
    isDenZeroReg_next(14) <= isDenZeroReg(13);
    isNegativeReg_next(13) <= isNegativeReg(12);
    isNumZeroReg_next(13) <= isNumZeroReg(12);
    isDenZeroReg_next(13) <= isDenZeroReg(12);
    isNegativeReg_next(12) <= isNegativeReg(11);
    isNumZeroReg_next(12) <= isNumZeroReg(11);
    isDenZeroReg_next(12) <= isDenZeroReg(11);
    isNegativeReg_next(11) <= isNegativeReg(10);
    isNumZeroReg_next(11) <= isNumZeroReg(10);
    isDenZeroReg_next(11) <= isDenZeroReg(10);
    isNegativeReg_next(10) <= isNegativeReg(9);
    isNumZeroReg_next(10) <= isNumZeroReg(9);
    isDenZeroReg_next(10) <= isDenZeroReg(9);
    isNegativeReg_next(9) <= isNegativeReg(8);
    isNumZeroReg_next(9) <= isNumZeroReg(8);
    isDenZeroReg_next(9) <= isDenZeroReg(8);
    isNegativeReg_next(8) <= isNegativeReg(7);
    isNumZeroReg_next(8) <= isNumZeroReg(7);
    isDenZeroReg_next(8) <= isDenZeroReg(7);
    isNegativeReg_next(7) <= isNegativeReg(6);
    isNumZeroReg_next(7) <= isNumZeroReg(6);
    isDenZeroReg_next(7) <= isDenZeroReg(6);
    isNegativeReg_next(6) <= isNegativeReg(5);
    isNumZeroReg_next(6) <= isNumZeroReg(5);
    isDenZeroReg_next(6) <= isDenZeroReg(5);
    isNegativeReg_next(5) <= isNegativeReg(4);
    isNumZeroReg_next(5) <= isNumZeroReg(4);
    isDenZeroReg_next(5) <= isDenZeroReg(4);
    isNegativeReg_next(4) <= isNegativeReg(3);
    isNumZeroReg_next(4) <= isNumZeroReg(3);
    isDenZeroReg_next(4) <= isDenZeroReg(3);
    isNegativeReg_next(3) <= isNegativeReg(2);
    isNumZeroReg_next(3) <= isNumZeroReg(2);
    isDenZeroReg_next(3) <= isDenZeroReg(2);
    isNegativeReg_next(2) <= isNegativeReg(1);
    isNumZeroReg_next(2) <= isNumZeroReg(1);
    isDenZeroReg_next(2) <= isDenZeroReg(1);
    isNegativeReg_next(1) <= isNegativeReg(0);
    isNumZeroReg_next(1) <= isNumZeroReg(0);
    isDenZeroReg_next(1) <= isDenZeroReg(0);
    IF isNumNegative /= isDenNegative THEN 
      isNegativeReg_next(0) <= '1';
    ELSE 
      isNegativeReg_next(0) <= '0';
    END IF;
    isNumZeroReg_next(0) <= isNumZero;
    isDenZeroReg_next(0) <= isDenZero;
    y_tmp <= zReg(16);
    IF isNumZeroReg(16) = '1' THEN 
      -- Negate with saturate so the most negative value doesn't overflow.
      y_tmp <= to_signed(16#0000#, 16);
    ELSIF isNegativeReg(16) = '1' THEN 
      -- Negate with saturate so the most negative value doesn't overflow.
      cast := resize(zReg(16), 17);
      cast_0 :=  - (cast);
      IF (cast_0(16) = '0') AND (cast_0(15) /= '0') THEN 
        y_tmp <= X"7FFF";
      ELSIF (cast_0(16) = '1') AND (cast_0(15) /= '1') THEN 
        y_tmp <= X"8000";
      ELSE 
        y_tmp <= cast_0(15 DOWNTO 0);
      END IF;
    END IF;
    t_tmp <= tReg(16);
    isDenZeroOut <= isDenZeroReg(16);
    validOut <= validReg(16);
  END PROCESS embreciprocals_c21_normalizedCORDICDivide_output;


  y <= std_logic_vector(y_tmp);

  t <= std_logic_vector(t_tmp);

END rtl;

