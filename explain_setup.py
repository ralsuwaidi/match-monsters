"""Show the repositioning decision on real board states."""
import random

import ai
import engine
import grid
import rules

NAMES = {'red': 'red', 'yellow': 'yellow', 'blue': 'blue',
         'green': 'green', 'purple': 'purple', grid.BERRY: 'berry'}


def tiles(d):
    return " + ".join(f"{n} {NAMES[t]}" for t, n in
                      sorted(d.items(), key=lambda x: -x[1]))


def main():
    rng = random.Random(4)
    g = grid.Grid(rng)
    me = engine.Side(engine.MY_TEAM, 80, 'me')
    foe = engine.Side(engine.FOE_TEAM, 85, 'foe')
    pol = ai.MY_POLICIES['bon_hdeny']
    shown = 0
    from collections import Counter
    st = Counter()

    for turn in range(60):
        ctx = ai.valuation(me, foe, pol)
        matches, setups = g.all_swaps(ctx.rank, want_setups=True)
        if matches and setups and shown < 4:
            best_sc, best_mv, best_cl = None, None, None
            for mv, cleared, longest in matches:
                sc = ai.score_move(me, foe, ctx, pol, cleared, longest, True)
                if best_sc is None or sc > best_sc:
                    best_sc, best_mv, best_cl = sc, mv, cleared
            threshold = best_sc + pol.setup_cost_w * max(best_sc, ctx.move_value)

            bestup, bestup_cl = None, None
            for mv, _pot in setups[:rules.SETUP_SEARCH_K]:
                clone = g.clone(rng=g.rng)
                clone.swap(mv)
                for _m2, cl2, ln2 in clone.moves_with_preview_near(
                        ((mv[0], mv[1]), (mv[2], mv[3]))):
                    sc2 = ai.score_move(me, foe, ctx, pol, cl2, ln2, True)
                    if bestup is None or sc2 > bestup:
                        bestup, bestup_cl = sc2, cl2
            shown += 1
            print(f"--- turn {turn + 1} " + "-" * 52)
            print(f"  best match available NOW      : {tiles(best_cl):<28} "
                  f"score {best_sc:6.1f}")
            print(f"  an ordinary move is worth     : {' ':<28} "
                  f"      {ctx.move_value:6.1f}")
            print(f"  so repositioning must unlock  : more than            "
                  f"        {threshold:6.1f}")
            if bestup_cl:
                print(f"  best it could actually unlock : {tiles(bestup_cl):<28} "
                      f"score {bestup:6.1f}")
            verdict = ("REPOSITION" if bestup and bestup > threshold
                       else "just take the match now")
            print(f"  -> {verdict}")
        engine.play_turn(me, foe, g, pol, st)
        engine.play_turn(foe, me, g, ai.FOE_POLICIES['pel_deny'], st)
        if me.hp <= 0 or foe.hp <= 0:
            break
    print()
    print(f"setup_cost_w = {pol.setup_cost_w:g}. The threshold is")
    print("    best match now  +  setup_cost_w x max(best match now, ordinary move)")
    print("so when a decent match is already on the board it works out at roughly")
    print("THREE TIMES the current best -- deliberately hard to clear.")


if __name__ == '__main__':
    main()
