"""Re-solve the matchup at each plausible berry spawn rate.

The berry rate is the one board parameter still unmeasured, and it changes which
strategy is best -- so the equilibrium has to be re-solved at every rate, not
just the win rate re-measured.
"""
from match_monsters import sim as sim

ME = ['bon_hdeny', 'bon_deny', 'bon_evo', 'no_berry']
FOE = ['pel_hdeny', 'pel_deny', 'pel_evo', 'no_berry']
RATES = [0.00, 0.04, 0.06, 0.08, 0.10, 0.125, 1 / 6]
TRIALS = 2500


def main():

    print(f"{'berry%':>7} {'equilibrium':>12} {'my plan':>11} {'their plan':>12}"
          f" {'berries/game':>13} {'my evo/game':>12}")
    for p in RATES:
        cfg = {'grid.BERRY_WEIGHT': p}
        M, S = {}, {}
        for r in ME:
            for c in FOE:
                res = sim.run(r, c, TRIALS, seed=31337, cfg=cfg)
                M[(r, c)] = 100 * res['wr']
                S[(r, c)] = res['st']
        my_best = max(ME, key=lambda r: min(M[(r, c)] for c in FOE))
        foe_best = min(FOE, key=lambda c: max(M[(r, c)] for r in ME))
        val = M[(my_best, foe_best)]
        st = S[(my_best, foe_best)]
        g = max(1, st['games'])
        print(f"{100*p:>6.1f}% {val:>11.1f}% {my_best:>11} {foe_best:>12}"
              f" {st['me/berries']/g:>13.1f} {st['me/evolve_Bonzumi']/g:>12.2f}")

    print("\nfull matrix at each rate, my win % (rows=me, cols=them):")
    for p in [0.06, 0.10, 1 / 6]:
        cfg = {'grid.BERRY_WEIGHT': p}
        print(f"\n  berry rate {100*p:.1f}%")
        print(f"    {'':<11}" + "".join(f"{c:>12}" for c in FOE))
        for r in ME:
            row = [100 * sim.run(r, c, TRIALS, seed=31337, cfg=cfg)['wr'] for c in FOE]
            print(f"    {r:<11}" + "".join(f"{v:>12.1f}" for v in row))


if __name__ == '__main__':
    main()
