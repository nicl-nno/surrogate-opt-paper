import glob
import os

import re

FORECAST_FILE_PATTERN = "K(\d)a_ns(\d+)_run(\d+)"


def forecast_files_from_dir(path):
    files = []
    for file in glob.iglob(os.path.join(path, '*.tab')):
        files.append(file)

    return files


def extracted_forecast_params(file_name, pattern=FORECAST_FILE_PATTERN):
    p = re.compile(pattern)
    match = p.search(file_name)

    return match.groups() if match is not None else ('', '', '')


def is_valid(forecast_file, expected_station, expected_noise_run):
    _, name = os.path.split(forecast_file)
    actual_station, actual_noise_run, _ = extracted_forecast_params(name)

    return True if actual_station == expected_station and actual_noise_run == expected_noise_run else False


def files_by_stations(files, noise_run, stations):
    groups = []
    for station in stations:
        groups.append(
            [file for file in files if is_valid(file, expected_station=station, expected_noise_run=str(noise_run))])

    return groups
