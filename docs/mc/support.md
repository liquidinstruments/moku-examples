# Moku Library

The Moku library is available to provide some useful types, functions and components
to use in your designs.  These design elements are defined in the Support package, and
can be used in your design like this:

```vhdl
library Moku;
use Moku.Support.clip;

architecture Behavioural of CustomWrapper is
    signal X : signed(31 downto 0);
begin
    -- Multiply results in bit growth to 32 bits
    X <= InputA * InputB;

    -- Use library function to "clip" the value to 16 bits
    OutputA <= clip(X, 15, 0);
end architecture;
```

:::warning Moku Library limits
The Moku Library currently supports VHDL and Verilog (SystemVerilog).
:::

## Moku.Support Functions

### sum_no_overflow

sum A and B and clip the result to the range of A instead of wrapping

```vhdl
function sum_no_overflow(A : signed; B : signed) return signed;
function sum_no_overflow(A : signed; B : integer) return signed;
function sum_no_overflow(A : unsigned; B : signed) return unsigned;
function sum_no_overflow(A : unsigned; B : integer) return unsigned;
```

### clip

Clip A from NewLeft to NewRight and saturate the result if original value
exceeds the resulting range

```vhdl
function clip(A : signed; NewLeft, NewRight : integer) return signed;
```

### clip_val

Clip A between MinVal and MaxVal (inclusive) without resizing the vector.

```vhdl
function clip_val(A : signed; MinVal, MaxVal : integer) return signed;
```

### or_reduce

Return the result of or'ing all bits in X

```vhdl
function or_reduce(X: std_logic_vector) return std_logic;
```

## Moku.Support Components

### ScaleOffset

`ScaleOffset` and `ScaleOffset2` are wrappers for a DSP block to compute `Z = X * Scale + Offset` and `Z = (X + Y) * Scale + Offset` respectively.

`Scale` can be up to 18 bits long and covers the range `-2^NORMAL_SHIFT -> 2^NORMAL_SHIFT`; i.e., `±1` with the default value of `NORMAL_SHIFT=0`. This means that if you need this block to scale *up*, then `NORMAL_SHIFT` must be set greater than 0; for example, if you need to scale up by a factor of 10x then `NORMAL_SHIFT=4` gives a range of `±16` then `Scale=0.625`, or `0x4FFF` in signed 16-bits, gives the required scale overall.

`OFFSET_SHIFT` gives the relative shift between the `Offset` field and the `X` argument. This is useful to get the `Offset` field to the same order as the `X * Scale` value.

`ROUNDING` is on by default. Set to `0` to floor the result.

The calculation can be registered at different locations to meet timing constraints. By default, the calculation is registered in the middle only, for a one clock cycle latency. This can be disabled by setting `MID_REG=0`. On the other hand, long timing paths to and from the block can be registered right at the input and/or output (`IN_REG=1` and/or `OUT_REG=1`) if required. Note that `IN_REG` and `MID_REG` are mutually exclusive and trying to set both at once will fail synthesis.

```vhdl
-- Z = X * Scale + Offset
component ScaleOffset
    generic (
        NORMAL_SHIFT : integer := 0;
        OFFSET_SHIFT : integer := 0;
        ROUNDING : boolean := true;
        IN_REG : integer range 0 to 1 := 0;
        OUT_REG : integer range 0 to 1 := 0;
        MID_REG : integer range 0 to 1 := 1
    );
    port (
        Clk : in std_logic;
        Reset : in std_logic;
        X : in signed;
        Scale : in signed;
        Offset : in signed;
        Z : out signed;
        Valid : in std_logic;
        OutValid : out std_logic
    );
end component;

-- Z = (X + Y) * Scale + Offset
component ScaleOffset2
    generic (
        NORMAL_SHIFT : integer := 0;
        OFFSET_SHIFT : integer := 0;
        ROUNDING : boolean := true;
        IN_REG : integer range 0 to 1 := 0;
        OUT_REG : integer range 0 to 1 := 0;
        MID_REG : integer range 0 to 1 := 1
    );
    port (
        Clk : in std_logic;
        Reset : in std_logic;
        X : in signed;
        Y : in signed;
        Scale : in signed;
        Offset : in signed;
        Z : out signed;
        Valid : in std_logic;
        OutValid : out std_logic
    );
end component;
```

### Interpolator

Linearly interpolate between A and B a distance of N
N is normalized between 0 and 1

```vhdl
component Interpolator
    generic (
        OUT_REG : integer range 0 to 1 := 0  -- optional output register
    );
    port (
        Clk : in std_logic;
        Reset : in std_logic;
        A : in signed;  --Point A
        B : in signed; --Point B
        N : in unsigned;  --Distance from A to B to resolve Z
        Z : out signed  --Result
    );
end component;
```

#### Example

```vhdl
library Moku;
use Moku.Support.Interpolator;

architecture Behavioural of CustomWrapper is
    signal A, B : signed(15 downto 0);
    signal N : unsigned(7 downto 0);
    signal Z : signed(15 downto 0);
begin
    -- These can be any dynamic signals
    A <= to_signed(0, 16);
    B <= to_signed(1000, 16);

    -- N is normalized to N'range, so 128 ~= 128 / 2^8 = 0.5
    N <= to_unsigned(128, 8);

    INTERP0: Interpolator
        port map (
            Clk => Clk,
            Reset => Reset,
            A => A,
            B => B,
            N => N,
            Z => Z
        );

    -- Z = (B - A) * 0.5 + A = 500
end architecture;
```
