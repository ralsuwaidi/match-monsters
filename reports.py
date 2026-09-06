"""The printed reports. Run via `python3 sim.py [board|matrix|head|sens|ai]`."""

import random
import sys
from collections import Counter

import grid
from grid import Grid
import monsters
import rules
from ai import MY_POLICIES, FOE_POLICIES, Policy, STRATEGIES_ME, STRATEGIES_FOE
from engine import Side
from monsters import MY_TEAM, FOE_TEAM
from runner import run, wilson


def report_board(trials=4000, seed=7):
    print("\n=== BOARD " + "=" * 58)
    print(f"{grid.W} wide x {grid.H} tall, {len(grid.ALL)} tile types at "
          f"{grid.BERRY_WEIGHT*100:.1f}% each, cascades={grid.CASCADE}\n")
    rng = random.Random(seed)
    avail = Counter()
    nmoves, nsetups, big, total = [], [], 0, 0
    for _ in range(trials):
        g = Grid(rng)
        m, s = g.all_swaps({t: 1.0 for t in grid.ALL})
        nmoves.append(len(m)); nsetups.append(len(s))
        seen = set()
        for mv, cl, ln in m:
            total += 1
            if ln >= 4:
                big += 1
            seen |= set(cl)
        for t in seen:
            avail[t] += 1
    print(f"matches available on a fresh board:  mean {sum(nmoves)/trials:.1f}"
          f"  (min {min(nmoves)}, max {max(nmoves)})")
    print(f"repositioning swaps worth considering: mean {sum(nsetups)/trials:.1f}")
    print(f"share of matches that are 4+ (earn an extra move): {100*big/total:.1f}%\n")
    print("chance a match of that tile type exists:")
    for t in grid.ALL:
        print(f"   {t:<8}{100*avail[t]/trials:>7.1f}%")

    # how long a board survives under greedy play before nothing is matchable
    lens = []
    for s in range(300):
        rr = random.Random(9000 + s)
        g = Grid(rr)
        n = 0
        for _ in range(400):
            m = g.moves_with_preview()
            if not m:
                break
            g.apply(rr.choice(m)[0]); n += 1
        lens.append(n)
    lens.sort()
    print(f"\nmatches playable before the board has nothing left to match: "
          f"median {lens[len(lens)//2]}")
    print("(a whole game is ~15 move-slots per side, so this is rarely reached)")


def report_matrix(trials=800, seed=99):
    print("\n=== STRATEGY MATRIX " + "=" * 48)
    print("my win rate (%), rows = my plan, cols = enemy plan\n")
    rows, cols = STRATEGIES_ME, STRATEGIES_FOE
    M = {(r, c): 100 * run(r, c, trials, seed=seed)['wr'] for r in rows for c in cols}
    print(f"{'':<12}" + "".join(f"{c:>12}" for c in cols) + f"{'WORST':>9}")
    for r in rows:
        v = [M[(r, c)] for c in cols]
        print(f"{r:<12}" + "".join(f"{x:>12.1f}" for x in v) + f"{min(v):>9.1f}")
    print(f"{'BEST':<12}" + "".join(f"{max(M[(r, c)] for r in rows):>12.1f}" for c in cols))
    maximin = max(min(M[(r, c)] for c in cols) for r in rows)
    minimax = min(max(M[(r, c)] for r in rows) for c in cols)
    my_best = max(rows, key=lambda r: min(M[(r, c)] for c in cols))
    foe_best = min(cols, key=lambda c: max(M[(r, c)] for r in rows))
    print(f"\nmy maximin (floor I can guarantee): {maximin:.1f}%  playing '{my_best}'")
    print(f"enemy minimax (ceiling they allow): {minimax:.1f}%  playing '{foe_best}'")
    print("-> saddle point: neither side gains by mixing" if abs(maximin - minimax) < 2.5
          else "-> no pure saddle; the true value sits between the two")
    return my_best, foe_best


def report_head(my_pol, foe_pol, trials=16000, seed=4242):
    print("\n=== HEADLINE " + "=" * 55)
    print(f"Bonzumi + Sipzap ['{my_pol}']   vs   Pelijet + Barbenin ['{foe_pol}']")
    res = run(my_pol, foe_pol, trials, seed=seed)
    p, n = res['wr'], res['n']
    lo, hi = wilson(p, n)
    print(f"\n  WIN RATE  {100*p:.1f}%   (95% CI {100*lo:.1f}-{100*hi:.1f}%, n={n})")
    print(f"    going first  (80 HP): {100*res['wr_first']:.1f}%")
    print(f"    going second (85 HP): {100*res['wr_second']:.1f}%")
    st = res['st']
    g = max(1, st['games'])
    print(f"\n  game length {st['turns']/g/2:.1f} rounds each; draws {st['draws']}")
    for lbl, team in (('me', MY_TEAM), ('foe', FOE_TEAM)):
        slots = max(1, st[f'{lbl}/move_slots'])
        print(f"\n  {lbl}")
        print("    damage/game  " + ", ".join(
            f"{m.name} {st[f'{lbl}/dmg_{m.name}']/g:.1f}" for m in team) +
            f"   total {sum(st[f'{lbl}/dmg_{m.name}'] for m in team)/g:.1f}")
        print("    strikes/game " + ", ".join(
            f"{m.name} {st[f'{lbl}/fires_{m.name}']/g:.2f}" for m in team))
        print("    mana/game    " + ", ".join(
            f"{m.color} {st[f'{lbl}/mana_{m.color}']/g:.1f}" for m in team) +
            f";  berries {st[f'{lbl}/berries']/g:.1f}")
        print("    evolutions   " + ", ".join(
            f"{m.name} {st[f'{lbl}/evolve_{m.name}']/g:.2f}" for m in team) +
            f";  boosts {st[f'{lbl}/boosts']/g:.2f}")
        print(f"    moves spent repositioning: {100*st[f'{lbl}/setup_moves']/slots:.0f}%"
              f";  extra moves earned/game {st[f'{lbl}/extra_moves']/g:.2f}")
        print("    colour matchable " + ", ".join(
            f"{m.color} {100*st[f'{lbl}/avail_{m.color}']/slots:.0f}%" for m in team))
    print(f"\n  mana Barbenin drains off me: {st['foe/drained']/g:.1f}/game")
    return p


def _sens_list():
    from monsters import honey_ohm, honey_ohm_convert, Ability
    return [
        ('baseline', {}),
        ('event HP: 70 not 80', {'BASE_HP': 70}),
        ('berry counter has no cap', {'BERRY_CAP': 99}),
        ('berry boost gives 2 mana not 4', {'BOOST_MANA': 2}),
        ('berry boost gives 6 mana not 4', {'BOOST_MANA': 6}),
        ('no cascades', {'grid.CASCADE': False}),
        ('berries rare (6%)', {'grid.BERRY_WEIGHT': 0.06}),
        ('no berries / no evolution', {'grid.BERRY_WEIGHT': 0.0}),
        ('evolving costs no move', {'EVOLVE_COSTS_MOVE': False}),
        ('mana does carry over', {'MANA_CARRYOVER': True}),
        ('Flare picks the best column', {'BEST_LINE': True}),
        ('abilities do not re-trigger', {'grid.RESOLVE_AFTER_ABILITY': False}),
        ('no repositioning swaps allowed', {'ALLOW_NON_MATCHING_SWAP': False}),
        ('Match Bonus mode (2 extra moves)', {'EXTRA_MOVES_PER_TURN': 2}),
        ('Barbenin drains 1 not 2/3', {'mon.BARBENIN.base': Ability(damage=10, drain=1),
                                       'mon.BARBENIN.evo': Ability(damage=15, drain=1)}),
        ('Barbenin does not drain', {'mon.BARBENIN.base': Ability(damage=10),
                                     'mon.BARBENIN.evo': Ability(damage=15)}),
        ('Bonzumi costs 7 not 8', {'mon.BONZUMI.cost': 7}),
        ('Bonzumi costs 6 not 8', {'mon.BONZUMI.cost': 6}),
    ]


def report_sens(my_pol, foe_pol, trials=6000, seed=555):
    print("\n=== SENSITIVITY " + "=" * 52)
    print("what the win rate does if one assumption is wrong\n")
    print(f"{'variant':<36}{'win %':>8}{'95% CI':>15}{'delta':>9}")
    base = None
    for name, cfg in _sens_list():
        res = run(my_pol, foe_pol, trials, seed=seed, cfg=cfg)
        p = res['wr']
        lo, hi = wilson(p, res['n'])
        d = '' if base is None else f"{100*(p-base):+.1f}"
        if base is None:
            base = p
        print(f"{name:<36}{100*p:>8.1f}{f'{100*lo:.1f}-{100*hi:.1f}':>15}{d:>9}")


def report_ai(my_pol, foe_pol, trials=6000, seed=808):
    print("\n=== AI STRENGTH " + "=" * 52)
    print("how much of the result is the players' skill rather than the rosters\n")
    for label, mp, fp in [
        ('both use repositioning', my_pol, foe_pol),
        ('I never reposition', 'bon_nosetup', foe_pol),
        ('they never reposition', my_pol, 'pel_nosetup'),
        ('neither repositions', 'bon_nosetup', 'pel_nosetup'),
        ('I reposition eagerly', 'bon_setup05', foe_pol),
        ('I reposition reluctantly', 'bon_setup15', foe_pol),
    ]:
        r = run(mp, fp, trials, seed=seed)['wr']
        lo, hi = wilson(r, trials)
        print(f"  {label:<28}{100*r:>7.1f}%   (CI {100*lo:.1f}-{100*hi:.1f})")


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    my_best, foe_best = 'bon_deny', 'pel_evo'
    if what in ('all', 'board'):
        report_board()
    if what in ('all', 'matrix'):
        my_best, foe_best = report_matrix()
    if what in ('all', 'head'):
        report_head(my_best, foe_best)
    if what in ('all', 'sens'):
        report_sens(my_best, foe_best)
    if what in ('all', 'ai'):
        report_ai(my_best, foe_best)


if __name__ == '__main__':
    main()
