# Custom Wrapper

The CustomWrapper entity defines the interface that custom designs need to
implement, as well as a simple abstraction from the instrument slot.

```vhdl
entity CustomWrapper is
    port (
        Clk : in std_logic;
        Reset : in std_logic;

        -- Input and Output use is platform-specific. These ports exist on all
        -- platforms but may not be externally connected.
        InputA : in signed(15 downto 0);
        InputB : in signed(15 downto 0);
        InputC : in signed(15 downto 0);
        InputD : in signed(15 downto 0);

        OutputA : out signed(15 downto 0);
        OutputB : out signed(15 downto 0);
        OutputC : out signed(15 downto 0);
        OutputD : out signed(15 downto 0);

        Control0 : in std_logic_vector(31 downto 0);
        Control1 : in std_logic_vector(31 downto 0);
        Control2 : in std_logic_vector(31 downto 0);
        Control3 : in std_logic_vector(31 downto 0);
        Control4 : in std_logic_vector(31 downto 0);
        Control5 : in std_logic_vector(31 downto 0);
        Control6 : in std_logic_vector(31 downto 0);
        Control7 : in std_logic_vector(31 downto 0);
        Control8 : in std_logic_vector(31 downto 0);
        Control9 : in std_logic_vector(31 downto 0)
    );
end entity;
```

Implementing the CustomWrapper interface simply requires defining an
architecture.

```vhdl
architecture Behavioural of CustomWrapper is
begin
    -- Add custom code here
end architecture;
```

::: warning
Only one architecture should implement CustomWrapper per project. If multiple
architectures exist, the one that is synthesized is undefined.
:::

## Wrapper Ports

The details of input, output and clock use is platform specific. For details, see [Input and Output](../mcc/io).

## Control Registers

These provide control of custom designs at runtime.
See [Control Registers](../mcc/controls).
