import numpy as np
import pytest

from customer_seg.predict import CLUSTER_LABELS, predict_customer_cluster


class DummyScaler:
    def transform(self, data):
        return np.array(data)


class DummyModel:
    def predict(self, data):
        return [0]


def test_predict_customer_cluster_returns_valid_cluster():
    model = DummyModel()
    scaler = DummyScaler()
    cluster = predict_customer_cluster(model, scaler, 10, 5, 100)
    assert cluster in CLUSTER_LABELS


@pytest.mark.parametrize(
    "value, name",
    [(-1, "Recency"), (-1, "Frequency"), (-1, "Monetary")],
)
def test_predict_customer_cluster_invalid_inputs(value, name):
    model = DummyModel()
    scaler = DummyScaler()
    with pytest.raises(ValueError):
        predict_customer_cluster(model, scaler, 10 if name != "Recency" else value, 5 if name != "Frequency" else value, 100 if name != "Monetary" else value)


def test_zero_rfm_values_are_valid():
    assert predict_customer_cluster(DummyModel(), DummyScaler(), 0, 0, 0) == 0


def test_complex_features_are_rejected_before_prediction():
    class ComplexScaler:
        def transform(self, data):
            return np.array([[1 + 2j, 3 + 0j, 4 + 0j]])

    with pytest.raises(ValueError, match="finite numeric features"):
        predict_customer_cluster(DummyModel(), ComplexScaler(), 10, 3, 40)
