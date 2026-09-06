"""Turn observed on-board berry counts into a spawn-rate estimate.

Counting berries on a mid-game board does NOT measure the spawn rate. Tiles
that get matched often are under-represented on the board; tiles that get
matched rarely pile up. Berries pile up especially hard, because once the
counter hits 4/4 a player stops matching them entirely.

So instead of assuming board count == spawn rate, this simulates real games at
each candidate rate and records what the board actually looks like mid-game.
"""
import random
import sys
from collections import Counter

import grid
import engine
import sim

OBSERVED = [3, 4, 4, 3, 5, 6]   # berries per 35-tile board, from screenshots
FRESH = [3]                     # boards seen at turn 1, before any matching
RATES = [0.06, 0.08, 0.10, 0.125, 1 / 6]


def main():
    print("board berry counts, mean of the screenshots: "
          f"{sum(OBSERVED)/len(OBSERVED):.2f}  (turn-1 boards: {FRESH})\n")
    print(f"{'spawn':>7} {'fresh board':>13} {'mid-game board':>16} {'when 4/4 held':>15}")
    for p in RATES:
        grid.BERRY_WEIGHT = p
        rng = random.Random(4242)

        fresh = [grid.Grid(rng).counts().get(grid.BERRY, 0) for _ in range(4000)]

        # mid-game: sample the board at the start of every turn of real duels
        mid, capped = [], []
        orig = engine.TRACE

        def spy(kind, **kw):
            if kind != 'turn_start':
                return
            n = kw['g'].counts().get(grid.BERRY, 0)
            mid.append(n)
            if kw['side'].berries >= sim.BERRY_CAP:
                capped.append(n)
        engine.TRACE = spy
        for i in range(400):
            sim.duel(rng, sim.MY_POLICIES['bon_hdeny'],
                     sim.FOE_POLICIES['pel_deny'], my_first=(i % 2 == 0))
        engine.TRACE = orig
        print(f"{100*p:>6.1f}% {sum(fresh)/len(fresh):>13.2f} "
              f"{sum(mid)/len(mid):>16.2f} "
              f"{(sum(capped)/len(capped) if capped else float('nan')):>15.2f}")

    print("\nRead the column that matches how the screenshots were taken.")
    print("Most were mid-game with a full 4/4 berry counter, which inflates the count.")


if __name__ == '__main__':
    main()
