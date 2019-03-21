from src.utils.files import (
    presented_fidelity,
    extracted_fidelity
)


def test_presented_fidelity():
    files = ['../path/to/fidelity/files/out_210_28km/1.tab',
             '../path/to/fidelity/files/out_210_56km/2.tab',
             '../path/to/fidelity/files/out_333_28km/213.tab',
             '../path/to/fidelity/files/out_110_56km/1.tab']

    actual_fid_time, actual_fid_space = presented_fidelity(files)

    assert len(actual_fid_time) == 3
    assert len(actual_fid_space) == 2


def test_extracted_fidelity():
    file = '../path/to/fidelity/files/out_210_28km/300.tab'

    actual_fid_time, actual_fid_space = extracted_fidelity(file)

    expected_fid_time, expected_fid_space = (210, 28)

    assert (actual_fid_time, actual_fid_space) == (expected_fid_time, expected_fid_space)
