import pytest

from customer_seg.loaders import LoaderError, load_data, load_models


def test_load_data_file_not_found(tmp_path):
    missing_file = tmp_path / "missing.csv"
    with pytest.raises(LoaderError):
        load_data(missing_file)


def test_load_models_files_not_found(tmp_path):
    with pytest.raises(LoaderError):
        load_models(tmp_path / "model.pkl", tmp_path / "scaler.pkl")


def test_load_data_invalid_utf8_raises_loader_error(tmp_path):
    corrupt_csv = tmp_path / "invalid.csv"
    corrupt_csv.write_bytes(b"Recency,Frequency,Monetary\n\xff,2,3\n")

    with pytest.raises(LoaderError, match="Unable to read customer CSV"):
        load_data(corrupt_csv)
