architecture Behavioural of CustomInstrument is
begin
    OutputA <= InputA when Control(1)(0) = '1' else (others => '0');
    OutputB <= InputB when Control(1)(1) = '1' else (others => '0');
end architecture;
