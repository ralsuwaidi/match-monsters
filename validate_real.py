"""Load the boards from the real-game screenshots and check them against the
engine's assumptions."""
import random
import grid
from grid import Grid

B, Y, G, P, R, S = 'blue', 'yellow', 'green', 'purple', 'red', grid.BERRY

SHOTS = {
 'screenshot 1 (my turn, 2 moves)': [
    [B, Y, G, P, R, B, P],
    [Y, S, G, B, Y, B, R],
    [G, P, Y, G, B, G, S],
    [Y, G, R, G, B, P, P],
    [P, S, Y, B, Y, P, R]],
 'screenshot 2 (after 1 move)': [
    [B, Y, R, G, P, B, P],
    [Y, S, G, P, P, B, R],
    [G, P, S, B, Y, G, S],
    [Y, G, B, Y, R, P, P],
    [P, S, R, G, R, P, R]],
 'game 2 turn 1 (70 vs 75 HP, all bars empty)': [
    [G, B, Y, R, R, Y, S],
    [B, G, P, P, R, B, R],
    [G, S, Y, B, P, G, Y],
    [P, G, B, G, Y, P, R],
    [P, B, R, Y, S, B, P]],
 'game 2 later (foe evolved, 45 vs 60 HP)': [
    [S, B, R, B, Y, P, S],
    [Y, R, R, B, S, P, R],
    [S, P, P, G, G, Y, R],
    [B, R, G, S, R, P, P],
    [G, G, R, Y, G, B, R]],
 'screenshot 3 (ZOZEZ to move)': [
    [B, Y, G, P, R, B, P],
    [Y, S, S, B, Y, B, R],
    [G, P, B, Y, B, G, S],
    [Y, G, R, G, B, P, P],
    [P, S, Y, B, Y, P, R]],
}

def load(rows):
    g = object.__new__(Grid)
    g.g = [r[:] for r in rows]
    g.rng = random.Random(0)
    return g

def main():
    for name, rows in SHOTS.items():
        g = load(rows)
        runs, longest = g.find_runs()
        m, s = g.all_swaps({t: 1.0 for t in grid.ALL})
        counts = g.counts()
        print(f"\n{name}")
        print(f"  already-matched tiles on board: {len(runs)}   <- must be 0")
        print(f"  legal matches available: {len(m)}   repositioning options: {len(s)}")
        print("  tile counts: " + ", ".join(f"{t} {counts.get(t,0)}" for t in grid.ALL)
              + f"   (total {sum(counts.values())})")
        by = {}
        for mv, cl, ln in m:
            for t in cl:
                by[t] = by.get(t, 0) + 1
        print("  colours you could match right now: "
              + (", ".join(f"{t}" for t in sorted(by)) or "none"))
        big = [ln for _, _, ln in m if ln >= 4]
        print(f"  matches of 4+ (would earn an extra move): {len(big)}")

    # --- how typical are these boards, per the engine? ---
    print("\n\nengine-generated boards, 20000 samples:")
    rng = random.Random(1)
    ms, per = [], {t: [] for t in grid.ALL}
    for _ in range(20000):
        gg = Grid(rng)
        ms.append(len(gg.moves_with_preview()))
        c = gg.counts()
        for t in grid.ALL:
            per[t].append(c.get(t, 0))
    ms.sort()
    print(f"  legal matches: mean {sum(ms)/len(ms):.1f}, "
          f"10th pct {ms[len(ms)//10]}, median {ms[len(ms)//2]}, "
          f"90th pct {ms[9*len(ms)//10]}")
    print("  tiles of each type: mean " +
          ", ".join(f"{t} {sum(per[t])/len(per[t]):.1f}" for t in grid.ALL))
    lo = {t: sorted(per[t])[len(per[t])//20] for t in grid.ALL}
    hi = {t: sorted(per[t])[19*len(per[t])//20] for t in grid.ALL}
    print("  90% of boards have each type between "
          f"{min(lo.values())} and {max(hi.values())} tiles")


if __name__ == '__main__':
    main()
