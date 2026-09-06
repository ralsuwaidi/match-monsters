"""Multiprocessing harness and statistics.

Config is passed explicitly to workers: macOS spawns rather than forks, so
module-level constants set in the parent would not survive.
"""

import math
import os
import random
from collections import Counter
from multiprocessing import Pool

import ai
import grid
import monsters
import rules
from ai import MY_POLICIES, FOE_POLICIES
from engine import duel


def _snapshot():
    """Pristine values, captured once per worker at import."""
    return (
        {k: getattr(rules, k) for k in dir(rules) if k.isupper()},
        {k: getattr(grid, k) for k in
         ('BERRY_WEIGHT', 'CASCADE', 'RESOLVE_AFTER_ABILITY')},
        {n: {f: getattr(m, f) for f in
             ('cost', 'base', 'evo', 'charge_base', 'charge_evo',
              'charge_cap', 'charge_cap_evo', 'charges_per_turn')}
         for n, m in vars(monsters).items()
         if isinstance(m, monsters.Monster)},
    )


_DEFAULTS = _snapshot()


def reset_defaults():
    """Undo any previous override. Needed because workers are now reused
    across cells, so one cell's sensitivity config must not leak into the next."""
    rl, gr, mo = _DEFAULTS
    for k, v in rl.items():
        setattr(rules, k, v)
    for k, v in gr.items():
        setattr(grid, k, v)
    for n, fields in mo.items():
        m = getattr(monsters, n)
        for f, v in fields.items():
            object.__setattr__(m, f, v)


def apply_cfg(cfg):
    """Apply a sensitivity override inside a worker.

    Keys are routed by prefix: `grid.X` -> the board engine, `mon.NAME.field`
    -> a monster's stats, anything else -> a rule constant. Because every
    module reads rules via `rules.X`, setting it here takes effect everywhere.
    """
    for k, v in (cfg or {}).items():
        if k.startswith('grid.'):
            setattr(grid, k[5:], v)
        elif k.startswith('mon.'):
            _, mname, field = k.split('.', 2)
            object.__setattr__(getattr(monsters, mname), field, v)
        else:
            if not hasattr(rules, k):
                raise KeyError(f"unknown rule constant {k!r}")
            setattr(rules, k, v)


def _run_block(args):
    my_name, foe_name, n, seed, mode, cfg = args
    reset_defaults()
    apply_cfg(cfg)
    # policies may arrive as a registered name or as a Policy built on the fly
    # by the search -- dataclasses pickle fine, so both work through the pool
    mp = MY_POLICIES[my_name] if isinstance(my_name, str) else my_name
    fp = FOE_POLICIES[foe_name] if isinstance(foe_name, str) else foe_name
    rng = random.Random(seed)
    st = Counter()
    tf = ts = 0.0
    nf = ns = 0
    for i in range(n):
        first = (i % 2 == 0) if mode == 'alt' else (mode == 'first')
        r = duel(rng, mp, fp, my_first=first, st=st)
        if first:
            tf += r; nf += 1
        else:
            ts += r; ns += 1
    return tf, nf, ts, ns, st


_POOL = {}


def get_pool(procs):
    """One long-lived pool. Spawning 10 fresh interpreters for every cell of a
    matrix costs more than the cell does."""
    p = _POOL.get(procs)
    if p is None:
        p = _POOL[procs] = Pool(procs)
    return p


def close_pool():
    for p in _POOL.values():
        p.terminate()
    _POOL.clear()


def run(my_name, foe_name, n, seed=12345, mode='alt', cfg=None, procs=None):
    procs = procs or min(os.cpu_count() or 4, 10)
    per = max(1, n // procs)
    jobs = [(my_name, foe_name, per, seed + 977 * i, mode, cfg) for i in range(procs)]
    if procs == 1:
        results = [_run_block(j) for j in jobs]
    else:
        results = list(get_pool(procs).imap_unordered(_run_block, jobs))
    st = Counter()
    tf = ts = 0.0
    nf = ns = 0
    for a, b, c, d, s in results:
        tf += a; nf += b; ts += c; ns += d; st.update(s)
    n_tot = nf + ns
    return {'wr': (tf + ts) / n_tot, 'n': n_tot,
            'wr_first': (tf / nf) if nf else None, 'n_first': nf,
            'wr_second': (ts / ns) if ns else None, 'n_second': ns, 'st': st}


def wilson(p, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(max(0.0, p * (1 - p) / n) + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)

