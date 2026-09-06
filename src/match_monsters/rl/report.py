"""Compact, pasteable summary of a training run.

    uv run --extra rl python training_report.py            # latest run
    uv run --extra rl python training_report.py sp-2026...  # a specific one

Prints everything needed to judge how a run went: the settings, the learning
curve, and the reference points that say what those numbers mean.
"""
import sys

from match_monsters.solver import progress

BASELINES = [
    ('uniform over legal moves', 24.7, 60.0, 10.2, 0),
    ('uniform over matches only', 141.5, 26.4, 95.0, 100),
    ('hand-tuned heuristic', 142.7, 16.0, 96.3, 100),
]


def pick_run(arg):
    runs = progress.list_runs()
    if arg:
        return arg if arg in runs else None
    for r in runs:
        s = progress.read(r)
        if s and s.get('kind') == 'neural':
            return r
    return runs[0] if runs else None


def main():
    rid = pick_run(sys.argv[1] if len(sys.argv) > 1 else None)
    if not rid:
        print('no runs found under runs/')
        return
    s = progress.read(rid)
    h = s.get('history', [])
    c = s.get('config', {})
    print('=' * 72)
    print(f'MATCH MONSTERS TRAINING REPORT   run {rid}')
    print('=' * 72)
    print(f'status      {s["status"]}   iteration {s.get("iter", 0)} of '
          f'{s.get("iters_planned", "?")}   device {s.get("device", "?")}')
    print(f'network     {c.get("arch")}  =  {c.get("params", 0):,} parameters'
          f'   (shared: {s.get("shared", False)})')
    print(f'rollout     {c.get("games")} parallel games, '
          f'{c.get("steps"):,} steps/iteration   lr {c.get("lr")}')
    print(f'rewards     shaping {c.get("shaping")}  entropy {c.get("entropy")}'
          f'  berry rate {c.get("berry_rate")}  board {c.get("board")}')
    if h:
        last = h[-1]
        print(f'progress    {last.get("steps", 0):,} steps in '
              f'{s.get("elapsed", 0)/3600:.2f} h  '
              f'({last.get("sps", 0):,.0f} steps/sec)')
    print()
    if not h:
        print('no iterations recorded yet')
        return

    print('LEARNING CURVE')
    print('%6s %7s %8s %8s %8s %8s %7s %7s %7s %8s'
          % ('iter', 'win%', 'turns', 'match%', 'damage', 'decisive',
             'evolves', 'fires', 'entropy', 'vs heur'))
    step = max(1, len(h) // 14)
    rows = h[::step]
    if h[-1] not in rows:
        rows.append(h[-1])
    for r in rows:
        ev = r.get('A_vs_heuristic')
        print('%6d %7.1f %8s %8.1f %8.1f %7.0f%% %7.2f %7.1f %7.2f %8s'
              % (r['iter'], r.get('wr') or float('nan'),
                 ('%.1f' % r['turns_to_win']) if r.get('turns_to_win') else '-',
                 r.get('match_rate', 0), r.get('damage', 0),
                 r.get('decisive', 0), r.get('evolutions', 0),
                 r.get('fires', 0), r.get('entropy_A') or 0,
                 ('%.0f%%' % ev) if ev is not None else '-'))
    print()
    print('REFERENCE POINTS  (what those numbers mean)')
    print('%-28s %8s %8s %8s %9s'
          % ('policy', 'damage', 'turns', 'match%', 'decisive'))
    for n, d, t, m, dec in BASELINES:
        print('%-28s %8.1f %8.1f %7.1f%% %8d%%' % (n, d, t, m, dec))
    last = h[-1]
    print('%-28s %8.1f %8s %7.1f%% %8.0f%%'
          % ('>> this run, latest', last.get('damage', 0),
             ('%.1f' % last['turns_to_win']) if last.get('turns_to_win') else '-',
             last.get('match_rate', 0), last.get('decisive', 0)))
    print()
    mr = last.get('match_rate', 0)
    if mr < 15:
        print('VERDICT: still at the random baseline (~10% match rate). The '
              'agent has not\n         learned to make matches at all.')
    elif mr < 60:
        print('VERDICT: learning has started -- match rate is above random but '
              'well short of\n         the 95% that random-over-matches '
              'achieves.')
    elif last.get('turns_to_win', 99) > 26:
        print('VERDICT: it makes matches reliably but plays them poorly -- '
              'turns-to-win is\n         worse than picking a random match. '
              'Strategy has not appeared yet.')
    else:
        print('VERDICT: genuinely competent. It is beating random-over-matches '
              'on speed,\n         which means it has learned strategy, not '
              'just matching.')
    print('=' * 72)


if __name__ == '__main__':
    main()
