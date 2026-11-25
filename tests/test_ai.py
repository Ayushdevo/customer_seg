import numpy as np
import pytest

from customer_seg.ai import AIServiceError, build_marketing_prompt, create_llm, generate_strategy


class DummyLLM:
    def __init__(self, content):
        self.content = content

    def invoke(self, prompt):
        return self


def test_build_marketing_prompt_contains_cluster_and_values():
    prompt = build_marketing_prompt(1, 30, 5, 200)
    assert "Cluster 1" in prompt
    assert "30 days" in prompt
    assert "$200.00" in prompt


@pytest.mark.parametrize("cluster", [True, np.bool_(True)])
def test_build_marketing_prompt_rejects_boolean_cluster(cluster):
    with pytest.raises(ValueError, match="Unknown customer cluster"):
        build_marketing_prompt(cluster, 30, 5, 200)


@pytest.mark.parametrize("cluster", [1.0, [], {}, 3])
def test_build_marketing_prompt_rejects_invalid_cluster_ids(cluster):
    with pytest.raises(ValueError, match="Unknown customer cluster"):
        build_marketing_prompt(cluster, 30, 5, 200)


def test_build_marketing_prompt_accepts_numpy_integer_cluster():
    assert "Cluster 1" in build_marketing_prompt(np.int64(1), 30, 5, 200)


def test_generate_strategy_returns_content():
    llm = DummyLLM("Test response")
    response = generate_strategy(llm, "prompt")
    assert response == "Test response"


def test_generate_strategy_raises_when_content_missing():
    class BadLLM:
        def invoke(self, prompt):
            return object()

    with pytest.raises(ValueError):
        generate_strategy(BadLLM(), "prompt")
