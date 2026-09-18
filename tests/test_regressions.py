import numpy as np
import pytest


def test_prediction_import_does_not_load_provider_sdk():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, '-c',
        "import customer_seg; import sys; assert 'langchain_google_genai' not in sys.modules"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('value', [None, True, np.bool_(False), '5', float('nan'), float('inf'), -float('inf')])
def test_numeric_validation_rejects_invalid_values(value):
    from customer_seg import validate_numeric_input
    with pytest.raises(ValueError, match='finite number'):
        validate_numeric_input(value, 'Recency')


@pytest.mark.parametrize('prediction', [[], [0, 1], [[0]], [.8], [np.nan], [True], ['0'], [99]])
def test_prediction_rejects_malformed_cluster_labels(prediction):
    from customer_seg import predict_customer_cluster
    class Scaler:
        def transform(self, values): return values
    class Model:
        def predict(self, values): return prediction
    with pytest.raises(ValueError):
        predict_customer_cluster(Model(), Scaler(), 1, 2, 3)


@pytest.mark.parametrize('scaled', [[[1, 2]], [[1, 2, np.inf]], [['a', 'b', 'c']], []])
def test_invalid_scaled_features_never_reach_model(scaled):
    from customer_seg import predict_customer_cluster
    class Scaler:
        def transform(self, values): return scaled
    class Model:
        def predict(self, values): raise AssertionError('must not execute')
    with pytest.raises(ValueError, match='Scaler must'):
        predict_customer_cluster(Model(), Scaler(), 1, 2, 3)


def test_prediction_respects_named_scaler_feature_order():
    from customer_seg import predict_customer_cluster
    class Scaler:
        feature_names_in_ = ['Monetary', 'Recency', 'Frequency']
        def transform(self, values):
            assert list(values.columns) == self.feature_names_in_
            assert values.iloc[0].tolist() == [300, 10, 5]
            return values.to_numpy()
    class Model:
        def predict(self, values): return [0]
    assert predict_customer_cluster(Model(), Scaler(), 10, 5, 300) == 0


def test_corrupt_artifacts_raise_loader_error(tmp_path):
    from customer_seg.loaders import load_models, LoaderError
    path = tmp_path / 'broken.pkl'
    path.write_text('not a serialized model')
    with pytest.raises(LoaderError, match='Unable to load'):
        load_models(path, path)


def test_loaders_reject_wrong_artifact_types(tmp_path, monkeypatch):
    from customer_seg import loaders
    path = tmp_path / 'model.pkl'
    path.touch()
    monkeypatch.setattr(loaders.joblib, 'load', lambda path: {})
    with pytest.raises(loaders.LoaderError, match='predict method'):
        loaders.load_models(path, path)
