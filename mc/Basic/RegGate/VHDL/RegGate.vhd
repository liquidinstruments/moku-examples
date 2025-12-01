architecture Behavioural of CustomWrapper is
begin
    OutputA <= InputA when Control1(0) = '1' else (others => '0');
    OutputB <= InputB when Control1(1) = '1' else (others => '0');
end architecture;
