import csv
import os
import time

import numpy as np


_METRICS_PATH = None
_SUMMARY_PATH = None
_SUMMARY_STEPS = set()
_WANDB_UPLOAD = False
_WRITTEN = set()

_HEADER = ['time', 'step', 'phase', 'metric', 'value']


def setup(metrics_path=None, summary_path=None, summary_steps='', wandb_upload=False):
    global _METRICS_PATH, _SUMMARY_PATH, _SUMMARY_STEPS, _WANDB_UPLOAD
    _METRICS_PATH = metrics_path or None
    _SUMMARY_PATH = summary_path or None
    _SUMMARY_STEPS = _parse_steps(summary_steps)
    _WANDB_UPLOAD = bool(wandb_upload)
    for path in (_METRICS_PATH, _SUMMARY_PATH):
        if path:
            _ensure_header(path)


def log_metrics(step, metrics, phase='training'):
    if _METRICS_PATH:
        _append_rows(_METRICS_PATH, step, phase, metrics)


def log_summary(step, metrics, phase='training', force=False):
    if not _SUMMARY_PATH:
        return
    if not force and int(step) not in _SUMMARY_STEPS:
        return
    key = (int(step), phase)
    if key in _WRITTEN:
        return
    _WRITTEN.add(key)
    _append_rows(_SUMMARY_PATH, step, phase, metrics)
    if _WANDB_UPLOAD:
        _upload(_METRICS_PATH)
        _upload(_SUMMARY_PATH)


def log_both(step, metrics, phase='training'):
    log_metrics(step, metrics, phase)
    log_summary(step, metrics, phase)


def _parse_steps(value):
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {int(v) for v in value}
    out = set()
    for item in str(value).split(','):
        item = item.strip()
        if item:
            out.add(int(item))
    return out


def _ensure_header(path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow(_HEADER)


def _append_rows(path, step, phase, metrics):
    _ensure_header(path)
    rows = []
    now = time.time()
    for key, value in sorted(_flatten(metrics).items()):
        value = _scalar(value)
        if value is not None:
            rows.append([now, int(step), phase, key, value])
    if rows:
        with open(path, 'a', newline='') as f:
            csv.writer(f).writerows(rows)


def _flatten(metrics, prefix=''):
    out = {}
    if metrics is None:
        return out
    for key, value in dict(metrics).items():
        name = f'{prefix}/{key}' if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten(value, name))
        else:
            out[name] = value
    return out


def _scalar(value):
    try:
        if isinstance(value, np.ndarray):
            if value.size != 1:
                return None
            value = value.reshape(()).item()
        elif hasattr(value, 'item'):
            value = value.item()
        if isinstance(value, (bool, np.bool_)):
            return float(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            value = float(value)
            if np.isfinite(value):
                return value
    except Exception:
        return None
    return None


def _upload(path):
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is not None and os.path.exists(path):
        wandb.save(path, policy='now')
