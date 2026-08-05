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
