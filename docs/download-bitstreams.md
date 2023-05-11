# Downloading bitstreams

Bitstreams comprise the instrument logic, and with each firmware update, a new set of bitstreams is released. To download the bitstreams using the `Python` client, you can simply execute the command `moku download --fw-ver=<fw_version>`. In `MATLAB`, you can use the function `moku_download(<fw_version>)` to accomplish the same.

If you are utilizing the REST interface directly or need to manually download the bitstreams for any other purpose, please follow the steps listed below:

1. Enter the firmware version in the provided field.
2. Click on the "Download Bitstreams" button to initiate the download of the bitstreams.
3. Additionally, you can click on the "Download Checksum" button to obtain the checksum file, which ensures the integrity of the downloaded content. To fetch the checksum of the downloaded tar file, you can run `md5sum mokudata-<fw_version>.tar.gz` and compare it against the downloaded checksum.

<bitstream-url></bitstream-url>

After downloading the file, you can extract its contents by unzipping the tar ball. Inside the extracted files, you will find directories that contain the bitstreams for various hardware platforms.

## MOKU_DATA_PATH

To manually download and provide bitstreams to Python and MATLAB clients, you can set the `MOKU_DATA_PATH` environment variable. This can be done by following these steps:

1. Determine the directory path where you have downloaded the bitstreams. Once you have the directory path, you can use it for both Python and MATLAB clients in the following ways:

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

Make sure to replace `/path/to/bitstreams/directory` with the actual path where the bitstreams are stored on your system.

By following these steps, you can ensure that both Python and MATLAB clients can access and utilize the bitstreams accordingly.
