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
