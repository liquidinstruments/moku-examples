library IEEE;
use IEEE.Std_Logic_1164.all;
use IEEE.Numeric_Std.all;


architecture Behavioural of CustomInstrumentInterlaced is
begin
    --gen_instance : for k in 0 to INPUT_INTERLACING_FACTOR generate
    --      ___ <= InputA(k);
    --      ___ <= InputB(k);
    --      ___ <= InputC(k);
    --      ___ <= InputD(k);
    --end generate gen_instance;
    
    -- ___ <= Control(0);
    -- ___ <= Control(1);
    -- ___ <= Control(2);
    --      ...
    -- ___ <= Control(15);
    
    --gen_instance : for k in 0 to OUTPUT_INTERLACING_FACTOR generate
    --      OutputA(k) => ___;
    --      OutputB(k) => ___;
    --      OutputC(k) => ___;
    --      OutputD(k) => ___;
    --end generate gen_instance;

    -- Status(1) => ___;
    -- Status(2) => ___;
    -- Status(3) => ___;
    --      ...
    -- Status(15) => ___;
    
end architecture;
