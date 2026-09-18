import numpy as np
import pytest


def test_prediction_import_does_not_load_provider_sdk():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, '-c',
        "import customer_seg; import sys; assert 'langchain_google_genai' not in sys.modules"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
