"""Shared run-state file, written by the solver and read by the UI.

The solver owns the file; the UI only ever reads it. Writes are atomic
(temp file + os.replace) so a reader never sees a half-written run.
"""
import json
import os
import time

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')


def run_dir(run_id):
    return os.path.join(RUNS_DIR, run_id)


def path(run_id):
    return os.path.join(run_dir(run_id), 'progress.json')


def log_path(run_id):
    return os.path.join(run_dir(run_id), 'solver.log')


def new_run_id():
    return time.strftime('%Y%m%d-%H%M%S')


def list_runs():
    if not os.path.isdir(RUNS_DIR):
        return []
    out = []
    for d in sorted(os.listdir(RUNS_DIR), reverse=True):
        if os.path.exists(path(d)):
            out.append(d)
    return out


def write(run_id, state):
    os.makedirs(run_dir(run_id), exist_ok=True)
    state['updated'] = time.time()
    tmp = path(run_id) + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f)
    os.replace(tmp, path(run_id))


def read(run_id):
    try:
        with open(path(run_id)) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True
