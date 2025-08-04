# mokucli convert

Convert Liquid Instruments binary data files (up to version 4) into formats like CSV and MATLAB

## Usage

```console
$ mokucli convert [OPTIONS] SOURCE
```

## Arguments

- `SOURCE`: Full path, relative path or filename of the .li file to convert [required]

## Options

- `--format [csv|npy|mat|hdf5]`: File format type to convert .li to [default: csv]
- `--help`: Show this message and exit

## Examples

```bash
# Convert .li file to CSV (default)
mokucli convert MokuDataLoggerData_20230114_142326.li

# Convert .li file to NumPy format
mokucli convert MokuDataLoggerData_20230114_142326.li --format npy

# Convert .li file to MATLAB format
mokucli convert MokuDataLoggerData_20230114_142326.li --format mat

# Convert .li file to HDF5 format
mokucli convert MokuDataLoggerData_20230114_142326.li --format hdf5
```

## Output

The converted file will be created in the same directory as the source file with the appropriate extension:
- `.csv` for CSV format
- `.npy` for NumPy format
- `.mat` for MATLAB format
- `.hdf5` for HDF5 format

## Notes

- This tool converts Liquid Instruments binary data files (.li) recorded by Moku devices
- The tool supports data files up to version 4 format
- Progress is shown during conversion with a progress bar
- The original .li file is preserved