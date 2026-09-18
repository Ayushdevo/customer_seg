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


@pytest.mark.parametrize('csv', ['wrong\n1\n', 'Recency,Frequency,Monetary\n', 'Recency,Frequency,Monetary\n1,2,inf\n', 'Recency,Frequency,Monetary\n1,-2,3\n'])
def test_customer_csv_schema_is_validated(tmp_path, csv):
    from customer_seg.loaders import load_data, LoaderError
    path = tmp_path / 'data.csv'
    path.write_text(csv)
    with pytest.raises(LoaderError):
        load_data(path)


def test_customer_csv_preserves_extra_columns(tmp_path):
    from customer_seg.loaders import load_data
    path = tmp_path / 'data.csv'
    path.write_text('Recency,Frequency,Monetary,Customer\n1,2,3,A\n')
    assert load_data(path).iloc[0]['Customer'] == 'A'


def test_marketing_prompt_rejects_invalid_metrics():
    from customer_seg import build_marketing_prompt
    with pytest.raises(ValueError):
        build_marketing_prompt(0, 1, 2, float('nan'))
    with pytest.raises(ValueError):
        build_marketing_prompt(99, 1, 2, 3)


def test_strategy_extracts_text_blocks_and_rejects_empty_content():
    from types import SimpleNamespace
    from customer_seg import generate_strategy, AIServiceError
    class LLM:
        def __init__(self, content): self.content = content
        def invoke(self, prompt): return SimpleNamespace(content=self.content)
    assert generate_strategy(LLM([{'type':'text', 'text':'Offer loyalty rewards'}, {'type':'image', 'url':'x'}]), 'prompt') == 'Offer loyalty rewards'
    for content in ['', '   ', [], None, [{'type':'image', 'url':'x'}]]:
        with pytest.raises(AIServiceError):
            generate_strategy(LLM(content), 'prompt')


def test_provider_failures_are_exposed_as_stable_service_errors():
    from customer_seg import generate_strategy, AIServiceError
    class LLM:
        def invoke(self, prompt): raise RuntimeError('provider-specific failure')
    with pytest.raises(AIServiceError) as result:
        generate_strategy(LLM(), 'Write a plan')
    assert isinstance(result.value.__cause__, RuntimeError)
    with pytest.raises(AIServiceError, match='prompt'):
        generate_strategy(LLM(), '  ')
