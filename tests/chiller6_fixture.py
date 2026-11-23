"""The original Chiller 6 conformal fixture (docs/calibration-batching.md)."""

import hashlib
import json
import os
from pathlib import Path

SOURCE_SHA256 = '8073af23ea71fc401aee97fd40b47678a80ad0549eb4153b1eb0ab8dc9b246e2'
VALUES_SHA256 = '3d2f4f8d3959eec792b212c457c1dfd5ace7683a76cbdb54754c8af554b1761a'
COLUMN = 'Chiller 6 Supply Temperature'
LENGTH = 2144  # 2,048 training observations and a 96-point holdout.
RELATIVE = Path('src/couchdb/scenarios_data/shared/iot/chiller_6.json')


def fixture_path():
    """``VTSFM_CHILLER6_JSON``, else the repository's data/AssetOpsBench copy,
    else the pinned checkout named by ``VTSFM_ASSETOPS_ROOT`` (same file)."""
    configured = os.environ.get('VTSFM_CHILLER6_JSON')
    if configured:
        return Path(configured)
    local = Path(__file__).resolve().parents[1] / 'data/AssetOpsBench' / RELATIVE
    checkout = os.environ.get('VTSFM_ASSETOPS_ROOT')
    if not local.is_file() and checkout:
        return Path(checkout) / RELATIVE
    return local


def load_series(path=None):
    """Sort by timestamp, drop duplicate timestamps and missing targets, keep the last 2,144."""
    import numpy as np
    import pandas as pd

    path = Path(path or fixture_path())
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError(f'{path} is not the pinned Chiller 6 fixture')
    frame = pd.DataFrame(json.loads(raw))
    frame['timestamp'] = pd.to_datetime(frame['timestamp'])
    frame = frame.sort_values('timestamp', kind='stable').drop_duplicates('timestamp')
    values = pd.to_numeric(frame[COLUMN], errors='coerce').dropna().to_numpy(dtype=np.float64)[-LENGTH:]
    if hashlib.sha256(values.tobytes()).hexdigest() != VALUES_SHA256:
        raise ValueError('Selected Chiller 6 values do not match the recorded fixture hash')
    return pd.Series(values, name=COLUMN)
