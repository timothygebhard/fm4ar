"""
Tests for `fm4ar.utils.hdf`.
"""

from pathlib import Path

import numpy as np

from fm4ar.utils.hdf import load_from_hdf, merge_hdf_files, save_to_hdf


def test__save_to_hdf__load_from_hdf(tmp_path: Path) -> None:
    """
    Test `save_to_hdf()` and `load_from_hdf()`.
    """

    file_path = tmp_path / "test.hdf"

    a1 = np.array([1, 2, 3])
    a2 = np.array([4.12, 5, 6 + 1 / 3])
    a3 = np.array([True, False, True])

    # Case 1: Save arrays
    save_to_hdf(file_path=file_path, a1=a1, a2=a2, a3=a3)

    # Case 2: Load arrays
    loaded = load_from_hdf(file_path=file_path, keys=["a1", "a2", "a3"])
    assert np.array_equal(loaded["a1"], a1)
    assert np.array_equal(loaded["a2"], a2)
    assert np.array_equal(loaded["a3"], a3)

    # Case 3: Load without specifying keys
    loaded = load_from_hdf(file_path=file_path, keys=None)
    assert set(loaded.keys()) == {"a1", "a2", "a3"}
    assert np.array_equal(loaded["a1"], a1)
    assert np.array_equal(loaded["a2"], a2)
    assert np.array_equal(loaded["a3"], a3)

    # Case 4: Load arrays with indices
    loaded = load_from_hdf(file_path=file_path, keys=["a1"], idx=1)
    assert np.array_equal(loaded["a1"], a1[1])


def test__merge_hdf_file(tmp_path: Path) -> None:
    """
    Test `merge_hdf_files()`.
    """

    # Create HDF5 files
    file1 = tmp_path / "file1.hdf"
    file2 = tmp_path / "file2.hdf"
    file3 = tmp_path / "file3.hdf"
    save_to_hdf(file_path=file1, a1=np.array([1, 2]), a2=np.array([3.0, 4.0]))
    save_to_hdf(file_path=file2, a1=np.array([5, 6]), a2=np.array([3.0, 4.0]))
    save_to_hdf(file_path=file3, a1=np.array([]), a2=np.array([]))

    # Merge HDF5 files
    output_file = tmp_path / "merged.hdf"
    merge_hdf_files(
        target_dir=tmp_path,
        name_pattern="file*.hdf",
        output_file_path=output_file,
        keys=["a1"],
        singleton_keys=["a2"],
        delete_after_merge=True,
    )

    # Load merged HDF5 file
    loaded = load_from_hdf(file_path=output_file, keys=["a1", "a2"])
    assert np.array_equal(loaded["a1"], np.array([1, 2, 5, 6]))
    assert np.array_equal(loaded["a2"], np.array([3.0, 4.0]))

    # Check if the original files were deleted
    assert not file1.exists()
    assert not file2.exists()
