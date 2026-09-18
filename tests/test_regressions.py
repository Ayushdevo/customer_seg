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
