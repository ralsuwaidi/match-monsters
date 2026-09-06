"""Co-evolutionary self-play: both sides improve until neither can.

Each side is a vector of scoring weights. A generation does one round of
iterated best response -- side A searches for the best answer to the pool of
side B's past champions, then B searches against A's pool. Evaluating against
a POOL rather than the single current champion is what stops the two sides
cycling (A beats B, B beats A, forever) instead of converging.

When neither side can improve any further, the head-to-head win rate that
remains is a property of the two ROSTERS, not of anyone's hand-tuning.

    uv run python coevolve.py --gens 40 --duels 800
"""
import argparse
import copy
import os
import random
import signal
import sys
import time
import traceback

import progress

_STOP = False


def _on_term(signum, frame):
    global _STOP
    _STOP = True


# continuous genes, with the range each is sampled and clipped to
GENES = {
    'deny_w':        (0.0, 3.0),
    'berry_w':       (0.0, 3.0),
    'neutral_w':     (0.0, 1.0),
    'extra_w':       (0.0, 3.0),
    'setup_cost_w':  (0.0, 4.0),
    'honey_patience': (0.0, 2.0),
}
MY_ORDERS = [('red',), ('yellow',), ('red', 'yellow'), ('yellow', 'red')]
FOE_ORDERS = [('blue',), ('purple',), ('blue', 'purple'), ('purple', 'blue')]


def genome_to_policy(gen, side, name):
    import ai
    return ai.Policy(name=name, own_w=1.0, use_berries=True,
                     evolve_order=gen['order'], setup=True,
                     **{k: gen[k] for k in GENES})


def rand_genome(rng, orders, seed_from=None):
    if seed_from is not None:
        g = copy.deepcopy(seed_from)
        return g
    g = {k: rng.uniform(lo, hi) for k, (lo, hi) in GENES.items()}
    g['order'] = rng.choice(orders)
    return g


def mutate(gen, rng, orders, sigma):
    g = copy.deepcopy(gen)
    for k, (lo, hi) in GENES.items():
        g[k] = min(hi, max(lo, g[k] + rng.gauss(0, sigma * (hi - lo))))
    if rng.random() < 0.15:
        g['order'] = rng.choice(orders)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', default=None)
    ap.add_argument('--gens', type=int, default=40)
    ap.add_argument('--duels', type=int, default=800, help='per evaluation')
    ap.add_argument('--lam', type=int, default=8, help='candidates per side per gen')
    ap.add_argument('--pool', type=int, default=6, help='past champions kept')
    ap.add_argument('--berry', type=float, default=None)
    ap.add_argument('--carryover', choices=['on', 'off'], default=None)
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    import ai
    import grid
    import rules
    from runner import run, wilson, close_pool

    cfg = {}
    if args.berry is not None:
        cfg['grid.BERRY_WEIGHT'] = args.berry
    if args.carryover is not None:
        cfg['MANA_CARRYOVER'] = (args.carryover == 'on')

    run_id = args.run_id or ('coev-' + progress.new_run_id())
    rng = random.Random(12345)

    # start both sides from the hand-tuned champions
    def from_policy(p, orders):
        g = {k: getattr(p, k) for k in GENES}
        g['order'] = p.evolve_order or (p.evolve_target,)
        if g['order'] not in orders:
            g['order'] = orders[0]
        return g

    me = from_policy(ai.MY_POLICIES['bon_hdeny'], MY_ORDERS)
    foe = from_policy(ai.FOE_POLICIES['pel_deny'], FOE_ORDERS)
    me_pool, foe_pool = [me], [foe]

    state = {
        'run_id': run_id, 'kind': 'coevolve', 'pid': os.getpid(),
        'started': time.time(), 'status': 'running', 'gen': 0,
        'gens_planned': args.gens, 'duels': 0,
        'config': {'berry_rate': cfg.get('grid.BERRY_WEIGHT', grid.BERRY_WEIGHT),
                   'mana_carryover': cfg.get('MANA_CARRYOVER', rules.MANA_CARRYOVER),
                   'base_hp': rules.BASE_HP, 'duels_per_eval': args.duels,
                   'lam': args.lam, 'pool': args.pool,
                   'board': f'{grid.W}x{grid.H}',
                   'second_player_hp_bonus': rules.SECOND_PLAYER_HP_BONUS,
                   'boost_mana': rules.BOOST_MANA, 'berry_cap': rules.BERRY_CAP,
                   'moves_per_turn': rules.MOVES_PER_TURN},
        'history': [], 'me': me, 'foe': foe, 'error': None,
    }
    progress.write(run_id, state)
    print(f'run {run_id} pid {os.getpid()}', flush=True)

    def score_me(gen, pool, seed):
        """Win rate of this genome against the opponent's whole pool."""
        p = genome_to_policy(gen, 'me', 'cand')
        tot = 0.0
        for o in pool:
            q = genome_to_policy(o, 'foe', 'opp')
            r = run(p, q, max(200, args.duels // len(pool)), seed=seed, cfg=cfg)
            tot += r['wr']
            state['duels'] += r['n']
        return tot / len(pool)

    def score_foe(gen, pool, seed):
        p = genome_to_policy(gen, 'foe', 'cand')
        tot = 0.0
        for o in pool:
            q = genome_to_policy(o, 'me', 'opp')
            r = run(q, p, max(200, args.duels // len(pool)), seed=seed, cfg=cfg)
            tot += r['wr']
            state['duels'] += r['n']
        return 1.0 - tot / len(pool)   # the foe wants MY win rate low

    try:
        sigma = 0.20
        for gen_i in range(1, args.gens + 1):
            if _STOP:
                raise KeyboardInterrupt
            seed = 4000 + gen_i          # common random numbers within a generation

            base = score_me(me, foe_pool, seed)
            best, best_s = me, base
            for _ in range(args.lam):
                cand = mutate(me, rng, MY_ORDERS, sigma)
                s = score_me(cand, foe_pool, seed)
                if s > best_s:
                    best, best_s = cand, s
            me_improved = best is not me
            me = best

            fbase = score_foe(foe, me_pool, seed)
            fbest, fbest_s = foe, fbase
            for _ in range(args.lam):
                cand = mutate(foe, rng, FOE_ORDERS, sigma)
                s = score_foe(cand, me_pool, seed)
                if s > fbest_s:
                    fbest, fbest_s = cand, s
            foe_improved = fbest is not foe
            foe = fbest

            me_pool = (me_pool + [me])[-args.pool:]
            foe_pool = (foe_pool + [foe])[-args.pool:]

            # the headline: current champions head to head, fresh sample
            h = run(genome_to_policy(me, 'me', 'champ'),
                    genome_to_policy(foe, 'foe', 'champ'),
                    args.duels * 3, seed=90000 + gen_i, cfg=cfg)
            state['duels'] += h['n']
            lo, hi = wilson(h['wr'], h['n'])
            state['history'].append({
                'gen': gen_i, 'wr': 100 * h['wr'], 'lo': 100 * lo, 'hi': 100 * hi,
                'n': h['n'], 'me_improved': me_improved, 'foe_improved': foe_improved,
                'me': {k: round(me[k], 3) for k in GENES} | {'order': list(me['order'])},
                'foe': {k: round(foe[k], 3) for k in GENES} | {'order': list(foe['order'])},
                'sigma': sigma,
            })
            state['gen'] = gen_i
            state['me'], state['foe'] = me, foe
            state['elapsed'] = time.time() - state['started']
            progress.write(run_id, state)
            print(f'gen {gen_i:3d}  win {100*h["wr"]:5.1f}%  '
                  f'me{"+" if me_improved else " "} foe{"+" if foe_improved else " "}  '
                  f'sigma {sigma:.3f}  duels {state["duels"]:,}', flush=True)

            # shrink the step when neither side found anything -- convergence
            if not me_improved and not foe_improved:
                sigma *= 0.75
                if sigma < 0.02:
                    print('converged: neither side can improve', flush=True)
                    break
            else:
                sigma = min(0.25, sigma * 1.05)

        state['status'] = 'done'
        progress.write(run_id, state)
    except KeyboardInterrupt:
        state['status'] = 'stopped'
        progress.write(run_id, state)
    except Exception:
        state['status'] = 'error'
        state['error'] = traceback.format_exc()
        progress.write(run_id, state)
        print(state['error'], flush=True)
        sys.exit(1)
    finally:
        close_pool()


if __name__ == '__main__':
    main()
