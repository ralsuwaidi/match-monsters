"""Play one duel and narrate it move by move.

    python3 trace.py [seed]
"""
import random
import sys

from match_monsters.game import grid
from match_monsters.game import engine
from match_monsters import sim as sim

GLYPH = {'red': 'R', 'yellow': 'Y', 'blue': 'B',
         'green': 'g', 'purple': 'P', grid.BERRY: '*'}
NAME = {'red': 'red', 'yellow': 'yellow', 'blue': 'blue',
        'green': 'green', 'purple': 'purple', grid.BERRY: 'berry'}


def who(side):
    return 'ME ' if side.label == 'me' else 'FOE'


def bars(side):
    out = []
    for c, m in side.mons.items():
        nm = m.name + ('+' if side.evolved[c] else '')
        b = f"{nm} {side.mana[c]}/{m.cost}"
        if m.charged:
            b += f" charge {side.charges[c]}/{m.cap(side.evolved[c])}"
        out.append(b)
    return "   ".join(out) + f"   berries {side.berries}/{sim.BERRY_CAP}"


def board(g):
    lines = ["      " + " ".join(f"{c}" for c in range(grid.W))]
    for r in range(grid.H):
        lines.append(f"    {r} " + " ".join(GLYPH[g.at(r, c)] for c in range(grid.W)))
    return "\n".join(lines)


def cells(mv):
    r, c, r2, c2 = mv
    return f"({r},{c})<->({r2},{c2})"


def tiles(d):
    return ", ".join(f"{n} {NAME[t]}" for t, n in
                     sorted(d.items(), key=lambda x: -x[1]))


STATE = {'turn': 0}


def printer(kind, **kw):
    side, foe, g = kw['side'], kw['foe'], kw['g']
    if kind == 'turn_start':
        STATE['turn'] += 1
        print(f"\n{'-'*72}")
        print(f"TURN {STATE['turn']}  --  {who(side)} to move       "
              f"HP  me {STATE['me'].hp}   foe {STATE['foe'].hp}")
        print(f"  {who(side)} {bars(side)}")
        print(board(g))
    elif kind == 'match':
        extra = "  +1 EXTRA MOVE" if kw['earned'] else ""
        print(f"  {who(side)} matches {cells(kw['mv'])}  ->  "
              f"{tiles(kw['cleared'])}   (run of {kw['longest']}){extra}")
    elif kind == 'setup':
        print(f"  {who(side)} repositions {cells(kw['mv'])} "
              f"-- no match, setting up the next move")
    elif kind == 'idle':
        print(f"  {who(side)} has nothing matchable, shuffles {cells(kw['mv'])}")
    elif kind == 'evolve':
        tgt = side.mons[kw['target']].name
        if kw['boost']:
            print(f"  {who(side)} spends a move + 4 berries to BOOST {tgt}")
        else:
            print(f"  {who(side)} spends a move + 4 berries to EVOLVE {tgt}")
    elif kind == 'fire':
        mon, ab = kw['mon'], kw['ab']
        bits = []
        if ab.damage:
            bits.append(f"{ab.damage} damage")
        if ab.clear_columns:
            bits.append(f"matches {ab.clear_columns} random column"
                        + ("s" if ab.clear_columns > 1 else ""))
        if ab.collect_n:
            bits.append(f"collects {ab.collect_n} random tiles")
        if ab.convert_n:
            bits.append(f"converts {ab.convert_n} tiles to {ab.convert_color}")
        if ab.drain:
            bits.append(f"drains {ab.drain} mana from EACH enemy monster")
        ch = f" at charge {kw['charges']}" if mon.charged else ""
        got = f"  picked up {tiles(kw['gained'])}" if kw['gained'] else ""
        print(f"  >> {mon.name} STRIKES{ch}: " + "; ".join(bits or ['nothing'])
              + got)
        print(f"     HP now  me {STATE['me'].hp}   foe {STATE['foe'].hp}")
    elif kind == 'turn_end':
        print(f"  end of turn: {who(side)} {bars(side)}")


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 20260905
    rng = random.Random(seed)

    real_side = sim.Side
    made = []

    class Spy(real_side):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)
            STATE[self.label] = self
    sim.Side = Spy
    engine.TRACE = printer
    print(f"MATCH MONSTERS -- one full game, seed {seed}")
    print("me  = Bonzumi (Fire, 8 mana) + Sipzap (Electric, 4 mana), 80 HP, moves first")
    print("foe = Pelijet (Water, 6 mana) + Barbenin (Psychic, 6 mana), 85 HP")
    print("board glyphs: R red  Y yellow  B blue  g green  P purple  * berry")
    r = sim.duel(rng, sim.MY_POLICIES['bon_hdeny'],
                 sim.FOE_POLICIES['pel_deny'], my_first=True)
    sim.Side = real_side
    print(f"\n{'='*72}")
    me, foe = STATE['me'], STATE['foe']
    print(f"RESULT: {'I WIN' if r == 1.0 else ('I LOSE' if r == 0.0 else 'DRAW')}"
          f"   final HP  me {me.hp}   foe {foe.hp}   after {STATE['turn']} turns")


if __name__ == '__main__':
    main()
