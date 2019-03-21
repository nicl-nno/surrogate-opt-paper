import csv
import glob
import os
import re
from datetime import datetime

PATH_TO_WW3_RESULTS = '../../samples/ww-res/'


class ObservationFile:
    def __init__(self, path, station_idx):
        self.path = path
        self.station_idx = station_idx

    def time_series(self, from_date="", to_date=""):
        '''
        Extract all wave heights from file with observations for a given time period
        :return: List of wave heights
        '''
        with open(os.path.join(os.path.dirname(__file__), self.path)) as file:
            lines = self._skip_meta_info(file.readlines())
            idx_from, idx_to = self._from_and_to_idxs(lines, from_date, to_date)
            waves = self._wave_heights(time_series=lines[idx_from:idx_to + 1])
            return waves

    def _skip_meta_info(self, lines):
        return list(filter(lambda line: line if not (line.startswith("#") or line.startswith("<")) else None, lines))

    def _from_and_to_idxs(self, lines, from_date="", to_date=""):
        idx_from, idx_to = -1, -1
        for line in lines:
            values = line.split()
            date, time = values[1], values[2]
            resulted_date = FormattedDate().target(date, time)
            if resulted_date == from_date:
                idx_from = lines.index(line)
            if resulted_date == to_date:
                idx_to = lines.index(line)

        assert idx_from < idx_to

        return idx_from, idx_to

    def _wave_heights(self, time_series):
        '''
        Extracting wave heights from time series of observation
        '''
        waves = [float(line.split()[4]) for line in time_series]
        return waves


class ForecastFile:
    def __init__(self, path):
        self.path = path

    def time_series(self):
        with open(self.path) as file:
            lines = self._skip_meta_info(file.readlines())
            return lines

    def _skip_meta_info(self, lines):
        return list(filter(lambda line: line if not line.startswith("V") else None, lines))


class FormattedDate:
    def __init__(self):
        self._source_date_pattern = "%d-%m-%Y %H:%M:%S"
        self._target_date_pattern = "%Y%m%d.%H"
        self._target_suffix = "0000"

    def target(self, date, time):
        return datetime.strptime(" ".join([date, time]), self._source_date_pattern).strftime(
            self._target_date_pattern) + self._target_suffix


class WaveWatchObservationFile:
    FILE_PATTERN = 'obs_fromww_([1-9]).csv'

    def __init__(self, path):
        self.path = path
        self.station_idx = self._parsed_station()

    def time_series(self, **kwargs):
        with open(os.path.join(os.path.dirname(__file__), self.path), newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            results = [float(row['hs']) for row in reader]

            return results

    def _parsed_station(self):
        _, name = os.path.split(self.path)
        p = re.compile(WaveWatchObservationFile.FILE_PATTERN)
        match = p.search(name)

        return match.groups()[0]


def observations_from_range(observations, range_values):
    # TODO: possible duplicate with Forecast._from_range()

    assert 0 <= range_values[0] <= range_values[1] <= 1

    from_idx = int(len(observations) * range_values[0])
    to_idx = int(len(observations) * range_values[1])

    return observations[from_idx:to_idx]


def real_obs_from_files():
    files = ["../../samples/obs/1a_waves.txt", "../../samples/obs/2a_waves.txt",
             "../../samples/obs/3a_waves.txt"]
    observations = []

    for station_idx, file in enumerate(files, 0):
        observations.append(ObservationFile(path=file, station_idx=station_idx))

    return observations


def wave_watch_results(path_to_results=PATH_TO_WW3_RESULTS, stations=None):
    '''

    :param path_to_results: Path to directory with ww3 results stored as csv-files
    :param stations: List of stations to take
    :return: List of WaveWatchObservationFiles objects according to chosen stations
    '''

    choice = '|'.join([str(station) for station in stations])
    file_pattern = f'obs_fromww_({choice}).csv'

    files = []
    for file in glob.iglob(os.path.join(path_to_results, '*.csv')):
        if re.search(file_pattern, file):
            files.append(file)

    result = [WaveWatchObservationFile(file) for file in sorted(files)]
    return result


FIDELITY_DIR_PATTERN = 'out_(\d+)_(\d+)km'


def presented_fidelity(files, fidelity_pattern=FIDELITY_DIR_PATTERN):
    p = re.compile(fidelity_pattern)

    fidelity_time = []
    fidelity_space = []

    for file in files:
        match = p.search(file)

        if match:
            fid_time = int(match.groups()[0])
            fid_space = int(match.groups()[1])

            if fid_time not in fidelity_time:
                fidelity_time.append(fid_time)
            if fid_space not in fidelity_space:
                fidelity_space.append(fid_space)

    return fidelity_time, fidelity_space


def extracted_fidelity(file, fidelity_pattern=FIDELITY_DIR_PATTERN):
    p = re.compile(fidelity_pattern)

    match = p.search(file)

    if match:
        fidelity_time = int(match.groups()[0])
        fidelity_space = int(match.groups()[1])

        return fidelity_time, fidelity_space

    else:
        return ''
