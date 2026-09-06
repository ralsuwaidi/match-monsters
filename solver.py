"""Detached policy solver.

Finds the policy with the best guaranteed floor -- the highest win rate against
the enemy's best reply -- and does it by racing rather than brute force.

Every policy starts with a small number of duels. After each round, any policy
whose floor cannot catch the current leader (its optimistic floor sits below the
leader's pessimistic floor) is dropped, and the duel budget doubles for the
survivors. Confidence intervals decide the eliminations, so a policy is only cut
when the evidence actually supports it. Duels accumulate across rounds -- nothing
measured is thrown away.

Runs fully detached; progress lands in runs/<id>/progress.json after every cell,
so a viewer can attach or detach without affecting it.

    python3 solver.py --start 400 --rounds 5 --berry 0.10 --carryover off
"""
import argparse
import os
import signal
import sys
import time
import traceback

import progress

_STOP = False


def _on_term(signum, frame):
    global _STOP
    _STOP = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', default=None)
    ap.add_argument('--start', type=int, default=400,
                    help='duels per cell in the first round')
    ap.add_argument('--rounds', type=int, default=6,
                    help='doubling rounds; the last is the most precise')
    ap.add_argument('--keep', type=int, default=2,
                    help='never race fewer than this many policies')
    ap.add_argument('--berry', type=float, default=None)
    ap.add_argument('--carryover', choices=['on', 'off'], default=None)
    ap.add_argument('--base-hp', type=int, default=None)
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    import grid
    import rules
    from ai import STRATEGIES_ME, STRATEGIES_FOE
    from runner import run, wilson, close_pool

    # Overrides must travel to the workers explicitly: they are spawned, not
    # forked, so they re-import this module's defaults from disk.
    cfg = {}
    if args.berry is not None:
        cfg['grid.BERRY_WEIGHT'] = args.berry
    if args.carryover is not None:
        cfg['MANA_CARRYOVER'] = (args.carryover == 'on')
    if args.base_hp is not None:
        cfg['BASE_HP'] = args.base_hp

    run_id = args.run_id or progress.new_run_id()
    ME, FOE = list(STRATEGIES_ME), list(STRATEGIES_FOE)

    state = {
        'run_id': run_id, 'pid': os.getpid(), 'started': time.time(),
        'status': 'running', 'phase': 'racing', 'round': 0,
        'me': ME, 'foe': FOE, 'alive': list(ME),
        'rounds_planned': args.rounds, 'start_trials': args.start,
        'config': {
            'berry_rate': cfg.get('grid.BERRY_WEIGHT', grid.BERRY_WEIGHT),
            'mana_carryover': cfg.get('MANA_CARRYOVER', rules.MANA_CARRYOVER),
            'base_hp': cfg.get('BASE_HP', rules.BASE_HP),
            'second_player_hp_bonus': rules.SECOND_PLAYER_HP_BONUS,
            'boost_mana': rules.BOOST_MANA, 'berry_cap': rules.BERRY_CAP,
            'moves_per_turn': rules.MOVES_PER_TURN,
            'board': f'{grid.W}x{grid.H}',
        },
        'cells': {}, 'eliminated': {}, 'duels': 0,
        'done': 0, 'total': len(ME) * len(FOE), 'best': None, 'error': None,
    }
    progress.write(run_id, state)
    print(f'run {run_id} started, pid {os.getpid()}', flush=True)

    acc = {}   # (me, foe) -> [wins, n, wins_first, n_first, wins_second, n_second, rounds_sum, rounds_n]

    def evaluate(r, c, n, seed):
        res = run(r, c, n, seed=seed, cfg=cfg)
        k = f'{r}|{c}'
        a = acc.setdefault(k, [0.0, 0, 0.0, 0, 0.0, 0, 0.0, 0])
        a[0] += res['wr'] * res['n']; a[1] += res['n']
        if res['wr_first'] is not None:
            a[2] += res['wr_first'] * res['n_first']; a[3] += res['n_first']
        if res['wr_second'] is not None:
            a[4] += res['wr_second'] * res['n_second']; a[5] += res['n_second']
        st = res['st']
        a[6] += st['turns'] / 2; a[7] += st['games']
        p = a[0] / a[1]
        lo, hi = wilson(p, a[1])
        state['cells'][k] = {
            'wr': 100 * p, 'n': a[1], 'lo': 100 * lo, 'hi': 100 * hi,
            'first': 100 * a[2] / a[3] if a[3] else None,
            'second': 100 * a[4] / a[5] if a[5] else None,
            'rounds': a[6] / max(1, a[7]),
        }
        state['duels'] += res['n']
        state['done'] += 1
        state['elapsed'] = time.time() - state['started']
        state['rate'] = state['duels'] / max(1e-9, state['elapsed'])
        progress.write(run_id, state)

    try:
        n = args.start
        for rnd in range(1, args.rounds + 1):
            if _STOP:
                raise KeyboardInterrupt
            alive = state['alive']
            state['round'] = rnd
            state['round_trials'] = n
            state['done'] = 0
            state['total'] = len(alive) * len(FOE)
            progress.write(run_id, state)

            for r in alive:
                for c in FOE:
                    if _STOP:
                        raise KeyboardInterrupt
                    evaluate(r, c, n, seed=1000 * rnd + 7)

            # floor = worst column. Compare optimistic floors against the best
            # pessimistic floor; only cut a policy that cannot catch up.
            pess = {r: min(state['cells'][f'{r}|{c}']['lo'] for c in FOE) for r in alive}
            opt = {r: min(state['cells'][f'{r}|{c}']['hi'] for c in FOE) for r in alive}
            leader = max(pess, key=lambda r: pess[r])
            bar = pess[leader]
            survivors = [r for r in alive if opt[r] >= bar]
            if len(survivors) < args.keep:
                survivors = sorted(alive, key=lambda r: -opt[r])[:args.keep]
            for r in alive:
                if r not in survivors:
                    state['eliminated'][r] = {
                        'round': rnd,
                        'floor': min(state['cells'][f'{r}|{c}']['wr'] for c in FOE),
                        'ceiling': opt[r], 'bar': bar, 'n': state['cells'][f'{r}|{FOE[0]}']['n'],
                    }
            state['alive'] = survivors
            state['floors'] = {r: min(state['cells'][f'{r}|{c}']['wr'] for c in FOE)
                               for r in alive}
            progress.write(run_id, state)
            print(f'round {rnd}: {n} duels/cell, {len(alive)} -> {len(survivors)} alive, '
                  f'leader {leader} floor {bar:.1f}%', flush=True)

            if len(survivors) == 1 and rnd >= 3:
                n *= 2
                continue
            n *= 2

        alive = state['alive']
        floors = {r: min(state['cells'][f'{r}|{c}']['wr'] for c in FOE) for r in alive}
        win = max(floors, key=lambda r: floors[r])
        reply = min(FOE, key=lambda c: state['cells'][f'{win}|{c}']['wr'])
        state['best'] = {'policy': win, 'floor': floors[win], 'enemy_reply': reply,
                         'cell': state['cells'][f'{win}|{reply}'], 'floors': floors}
        state['phase'] = 'done'
        state['status'] = 'done'
        progress.write(run_id, state)
        print(f'best: {win} floor {floors[win]:.1f}% vs {reply} '
              f'({state["duels"]} duels total)', flush=True)
    except KeyboardInterrupt:
        state['status'] = 'stopped'; state['phase'] = 'stopped'
        progress.write(run_id, state)
        print('stopped', flush=True)
    except Exception:
        state['status'] = 'error'; state['error'] = traceback.format_exc()
        progress.write(run_id, state)
        print(state['error'], flush=True)
        sys.exit(1)
    finally:
        close_pool()


if __name__ == '__main__':
    main()
