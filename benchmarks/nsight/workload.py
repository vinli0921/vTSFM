"""Inputs shared by the Nsight drivers: profiling-baseline harness data and windows.

Copied from `profiling/harness.py` on branch `profiling-baseline` (load_series, windows) so the
issue #16 rounds use the same series and windows without that branch. The AssetOpsBench checkout
is located with AOB_ROOT; only `chiller_6.json` is read.
"""
import json
import os
from pathlib import Path

import numpy as np

HORIZON = 96
TARGET = 'Chiller 6 Supply Temperature'
LEVELS = [q / 10 for q in range(1, 10)]


def data_path():
    root = Path(os.environ.get('AOB_ROOT', 'AssetOpsBench'))
    return root / 'src/couchdb/scenarios_data/shared/iot/chiller_6.json'


def load_series(target=TARGET):
    """Target column in time order; rows without a value are dropped, not interpolated."""
    import pandas as pd
    df = pd.DataFrame(json.loads(data_path().read_text()))
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
    df = df.sort_values('timestamp').drop_duplicates('timestamp')
    return df[target].dropna().astype(float).to_numpy()


def windows(y, batch, context, *, stride=None, offset=0):
    """`batch` (context, future) pairs; the series is tiled when it is too short."""
    need = context + HORIZON + (batch - 1) * (stride or 1) + offset
    reps = int(np.ceil(need / len(y)))
    ext = np.tile(y, reps) if reps > 1 else y
    room = len(ext) - context - HORIZON - offset
    stride = stride or max(1, room // max(batch, 1))
    ends = [context + offset + i * stride for i in range(batch)]
    ctx = np.stack([ext[e - context:e] for e in ends]).astype(np.float32)
    fut = np.stack([ext[e:e + HORIZON] for e in ends]).astype(np.float32)
    return ctx, fut, bool(context + HORIZON + offset + (batch - 1) * stride > len(y))


def round_inputs(y, round_name, patch):
    """Callable i -> context array [rows, 2048] for harness round 1, 0b, 0b16, or 0b8."""
    if round_name == '1':  # batch 1; the window advances one model patch per request
        return lambda i: windows(y, 1, 2048, stride=1, offset=i * patch)[0]
    rows = {'0b': 64, '0b16': 16, '0b8': 8}[round_name]
    fixed = windows(y, rows, 2048)[0]
    return lambda i: fixed
