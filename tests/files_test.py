from src.utils.files import presented_fidelity


def test_presented_fidelity():
    files = ['../path/to/fidelity/files/out_210/1.tab',
             '../path/to/fidelity/files/out_210/2.tab',
             '../path/to/fidelity/files/out_333/213.tab',
             '../path/to/fidelity/files/out_110/1.tab']

    actual_fidelity = presented_fidelity(files)

    assert len(actual_fidelity) == 3
