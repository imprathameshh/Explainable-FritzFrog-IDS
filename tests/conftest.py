import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_frame():
    """Synthetic frame mimicking CIC-IDS2018 quirks.

    Row layout (before cleaning):
      - index 0 and 1 are exact duplicates              -> 1 removed
      - index 2 has a NaN in "Flow Bytes/s"             -> dropped
      - index 3 has +inf in "Flow Duration"             -> dropped (via inf->NaN)
      - column " Dst Port " carries whitespace          -> stripped
      - "Timestamp" is an identifier column             -> dropped when present

    Net result: 5 rows -> 2 rows, "Timestamp" column removed.
    """
    return pd.DataFrame(
        {
            " Dst Port ": [80, 80, 443, 22, 8080],
            "Flow Duration": [100, 100, 200, np.inf, 300],
            "Flow Bytes/s": [1.0, 1.0, np.nan, 2.0, 3.0],
            "Timestamp": ["t0", "t0", "t1", "t2", "t3"],
            "Label": ["Benign", "Benign", "Bot", "Bot", "Benign"],
        }
    )
