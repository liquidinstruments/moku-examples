# Control Registers

The CustomWrapper provides 10 Control registers which can be used to control
the behaviour of the custom design at runtime. The registers are labelled
**Control0** through to **Control9** and are all 32 bit std_logic_vectors.

## Type Casting

These Controls can be assigned to various signals in a custom design and will
often require casting to another type or resizing or both.

```vhdl
-- Import libraries that contain the types we need
library IEEE;
use IEEE.Std_Logic_1164.all;  -- for std_logic(_vector) and resize()
use IEEE.Numeric_Std.all;  -- for signed and unsigned etc.

architecture Behavioural of CustomWrapper is
    -- define signals here
    signal A : signed(12 downto 0);
    signal B : std_logic;
    signal C : unsigned(63 downto 0);
begin

    A <= signed(Control0(12 downto 0));  -- take 13 LSBs and cast to signed
    B <= Control0(15);  -- Controls can be shared
    -- resize Control1 to 64 bits, MSBs padded with '0'
    C <= resize(unsigned(Control1), C'length);

end architecture;
```
