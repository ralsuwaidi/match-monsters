"""Answer a 'which line is better here?' question from the live game position.

Takes the saved board, plays THIS turn under two different policies, then plays
the rest of the game identically under both. The difference is the value of the
decision, not of the policy.
"""
import copy
import random
import sys
from collections import Counter

from match_monsters.agents import ai
from match_monsters.game import engine
from match_monsters.game import grid
from match_monsters import play as play
from match_monsters import rules

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
CONTINUE = 'bon_hdeny'
FOE = 'pel_deny'


def clone_state(g, me, foe, seed):
    g2 = object.__new__(grid.Grid)
    g2.g = [row[:] for row in g.g]
    g2.rng = random.Random(seed)
    g2.last_longest = 0
    return g2, copy.deepcopy(me), copy.deepcopy(foe)


def rollout(g0, me0, foe0, first_turn_policy, seed):
    g, me, foe = clone_state(g0, me0, foe0, seed)
    st = Counter()
    engine.play_turn(me, foe, g, ai.MY_POLICIES[first_turn_policy], st)
    if foe.hp <= 0:
        return 1.0
    for _ in range(rules.MAX_TURNS):
        engine.play_turn(foe, me, g, ai.FOE_POLICIES[FOE], st)
        if me.hp <= 0:
            return 0.0
        engine.play_turn(me, foe, g, ai.MY_POLICIES[CONTINUE], st)
        if foe.hp <= 0:
            return 1.0
    return 0.5


def main():
    g, me, foe, turn_no, mover, moves, extras = play.load()
    print(f'from the live position (turn {turn_no}, your move)')
    print(f'  you  hp {me.hp}  Bonzumi {me.mana["red"]}/8  '
          f'Ranzap {me.mana["yellow"]}/4 honey {me.charges["yellow"]}/3  '
          f'berries {me.berries}/4')
    print(f'  foe  hp {foe.hp}  Pelijet {foe.mana["blue"]}/6  '
          f'Barbenin {foe.mana["purple"]}/6  berries {foe.berries}/4')
    print(f'\n{N} rollouts per line, identical play after this turn\n')
    for label, pol in (('BUILD  (ignore their tiles)', 'line_build'),
                       ('DENY   (strip their tiles)', 'line_deny'),
                       ('BALANCE(the solver default)', 'bon_hdeny')):
        wins = sum(rollout(g, me, foe, pol, 90000 + i) for i in range(N))
        p = wins / N
        lo, hi = __import__('runner').wilson(p, N)
        print(f'  {label}   win {100*p:5.1f}%   (95% CI {100*lo:.1f}-{100*hi:.1f})')


if __name__ == '__main__':
    main()
