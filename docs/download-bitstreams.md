# Downloading bitstreams

Bitstreams contain the instrument logic and are released with each firmware update. To download the bitstreams using the **Python** client, simply run the command `moku download --fw-ver=<fw_version>`. In **MATLAB**, you can use the function `moku_download(<fw_version>)` to achieve the same.

If you need to manually download the bitstreams or utilize the REST interface directly, follow the steps below:

- Enter the firmware version in the provided field.
- Click the "Download Bitstreams" button to initiate the download.

<bitstream-url></bitstream-url>

Once the file is downloaded, you can extract its contents by unzipping the tarball. Inside the extracted files, you will find directories containing the bitstreams for different hardware platforms.

## MOKU_DATA_PATH

To manually download and provide bitstreams to Python and MATLAB clients, you can set the `MOKU_DATA_PATH` environment variable. 

Determine the directory path where you have downloaded the bitstreams. Once you have the directory path, you can use it for both Python and MATLAB clients in the following ways:

For Python:

- Simply provide the directory path as is to the Python client.

  - For Linux/macOS:

    ```shell
    export MOKU_DATA_PATH=/path/to/bitstreams/directory
    ```

  - For Windows (Command Prompt):

    ```shell
    set MOKU_DATA_PATH=C:\path\to\bitstreams\directory
    ```

  - For Windows (PowerShell):
    ```powershell
    $env:MOKU_DATA_PATH = "C:\path\to\bitstreams\directory"
    ```

For MATLAB:

- Locate the downloaded tar ball file on your system.
- Extract the contents of the tar ball to a directory of your choice.
- Note down the path of the extracted directory. The extracted directory will have a similar structure to `/path/to/extracted/bitstreams/mokudata-<fw_version>`

  - In MATLAB Command Window:
    ```shell
    setenv('MOKU_DATA_PATH', '/path/to/extracted/bitstreams/mokudata-<fw_version>')
    ```
