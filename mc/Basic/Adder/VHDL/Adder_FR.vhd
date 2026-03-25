architecture Behavioural of CustomInstrumentInterlaced is
begin
    gen1 : for k in 0 to INPUT_INTERLACING_FACTOR-1 generate
        -- Assign sum of inputs A and B to OutputA
        OutputA(k) <= InputA(k) + InputB(k);

        -- Assign difference of inputs A and B to OutputB
        OutputB(k) <= InputA(k) - InputB(k);
    end generate gen1;
end architecture;
