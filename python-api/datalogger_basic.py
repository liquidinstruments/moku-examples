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

from moku.instruments import Datalogger

# Replace with your Moku's IP address.
DEVICE_IP = '192.168.###.###'

# Connect to the Moku (force_connect will overtake an existing connection).
with Datalogger(DEVICE_IP, force_connect=True) as i:
    # Configure the frontend
    i.set_frontend(channel=1, impedance='1MOhm', coupling="AC", range="10Vpp")

    i.set_acquisition_mode(mode='Precision')

    # Generate Sine wave on Output1
    i.generate_waveform(channel=1, type='Sine', amplitude=1, frequency=10e3)

    # Log 100 samples per second
    # Stop an existing log, if any, then start a new one. 10 seconds of both
    # channels
    log_file = i.start_logging(duration=10, sample_rate=100)

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

    # Download log from Moku, use liconverter to convert this .li file to .csv
    i.download("persist", log_file['file_name'], os.path.join(os.getcwd(), log_file['file_name']))
    print("Downloaded log file to local directory.")
