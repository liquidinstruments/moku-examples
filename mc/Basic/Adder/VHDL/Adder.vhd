-- A very simple example, simply add two inputs and route to an output.
-- This is purely combinatorial
architecture Behavioural of CustomInstrument is
begin
    OutputA <= InputA + InputB;
    OutputB <= InputA - InputB;
end architecture;
