#
# moku example: Basic Datalogger
#
# This example demonstrates use of the Datalogger instrument to log time-series
# voltage data to a (Binary or CSV) file.
#
# (c) Liquid Instruments Pty. Ltd.
#
import os
import time
import subprocess
from moku.instruments import Datalogger

def convert_li(input_file, output_format="csv"):
    """
    Converts a  .li file to csv/npy/mat/hdf5 using mokucli.
    
    Args:
        input_file (str or Path): Path of .li file
        output_format (str): 'csv', 'npy', 'mat', o 'hdf5'
    """
    input_file = pathlib.Path(input_file).resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"File non trovato: {input_file}")

    # mokucli command
    cmd = [
        "mokucli",
        "convert",
        str(input_file),
        "--format", output_format
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Conversion error:\n{result.stderr}")


# Connect to your Moku by its ip address using Datalogger('192.168.###.###')
# force_connect will overtake an existing connection
i = Datalogger('192.168.###.###', force_connect=True)

try:
    # Configure the frontend
    i.set_frontend(channel=1, impedance='1MOhm', coupling="AC", range="10Vpp")

    i.set_acquisition_mode(mode='Precision')

    # Generate Sine wave on Output1
    i.generate_waveform(channel=1, type='Sine', amplitude=1, frequency=10e3)

    # Log 100 samples per second
    # Stop an existing log, if any, then start a new one. 10 seconds of both
    # channels
    logFile = i.start_logging(duration=10, sample_rate=100)

    # Track progress percentage of the data logging session
    complete = False
    while complete is False:
        # Wait for the logging session to progress by sleeping 0.5sec
        time.sleep(0.5)
        # Get current progress percentage and print it out
        progress = i.logging_progress()
        complete = progress['complete']
        if 'time_remaining' in progress:
            print(f"Remaining time {progress['time_remaining']} seconds")

    # Download log from Moku
    i.download("persist", logFile['file_name'], os.path.join(os.getcwd(), logFile['file_name']))
    # Converts file in place to 'npy'
    convert_li( os.path.join(os.getcwd(), logFile['file_name']), output_format = 'npy')
    print("Downloaded log file to local directory.")

except Exception as e:
    i.relinquish_ownership()
    raise e
finally:
    # Close the connection to the Moku device
    # This ensures network resources and released correctly
    i.relinquish_ownership()
