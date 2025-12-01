# IP Core Template

This folder contains the instantiation templates for including the Moku library of pre-compiled IP cores. The different IP core are listed in [this documentation](https://apis.liquidinstruments.com/mcc/ipcore.html#pre-compiled-ip-cores), including an example of using them in a custom code. 

The templates are provided as .vho (compatible with VHDL scripts) and as .veo (compatible with Verilog scripts), along with information on the ports used for the IP core. To use the instantiation template, open either .vho or .veo file of the desired IP core, and copy the module or the component into the custom script. the code. This information is available as a snippet within the files as indicated below. 

## VHDL (.vho)

<code-group>

```vhdl
------------- Begin Cut here for COMPONENT Declaration ------ COMP_TAG
        COMPONENT
            .
            .
            .
            .
            .

-- COMP_TAG_END ------ End COMPONENT Declaration ------------

-- The following code must appear in the VHDL architecture
-- body. Substitute your own instance name and net names.

------------- Begin Cut here for INSTANTIATION Template ----- INST_TAG
        PORT MAP
            .
            .
            .
            .

-- INST_TAG_END ------ End INSTANTIATION Template ---------
```

</code-group>

## Verilog (.veo)

<code-group>

```verilog
//----------- Begin Cut here for INSTANTIATION Template ---// INST_TAG

        module
            .
            .
            .
            .
            .

// INST_TAG_END ------ End INSTANTIATION Template --------- 
```
</code-group>