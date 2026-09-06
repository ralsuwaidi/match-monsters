"""Reference points for the learning curve: what does each level of skill look
like? Run this to know what a training number actually means."""
import random

import numpy as np

from match_monsters.agents import ai
from match_monsters.game import engine
from match_monsters.rl.selfplay import Duel, SWAP_IX


def run(pick, n=300, seed=4):
    rng = random.Random(seed)
    teams = (engine.MY_TEAM, engine.FOE_TEAM)
    dmg, turns, mrate, dec = [], [], [], 0
    for _ in range(n):
        d = Duel(random.Random(rng.randrange(1 << 30)), teams, max_turns=60)
        while not d.done:
            d.step(pick(d))
        dmg.append(sum(v for k, v in d.st.items() if '/dmg_' in k))
        turns.append(d.turn)
        mrate.append(d.n_matched / max(1, d.n_swaps))
        dec += 1 if d.result in (0.0, 1.0) else 0
    return np.mean(dmg), np.mean(turns), 100 * np.mean(mrate), 100 * dec / n


def uniform_legal(d):
    return int(np.random.choice(np.flatnonzero(d.legal_mask())))


def matches_only(d):
    m, _ = d.g.all_swaps(None, want_setups=False)
    return uniform_legal(d) if not m else SWAP_IX[m[np.random.randint(len(m))][0]]


def heuristic(d):
    from match_monsters.rl import evaluate as eval_nn
    pol = (ai.MY_POLICIES['bon_hdeny'] if d.active == 0
           else ai.FOE_POLICIES['pel_deny'])
    return eval_nn.HeuristicAgent(pol)._one(d)


def main():
        print('%-28s %9s %8s %12s %10s'
              % ('policy', 'damage', 'turns', 'match rate', 'decisive'))
        for name, f in (('uniform over legal moves', uniform_legal),
                        ('uniform over MATCHES only', matches_only),
                        ('hand-tuned heuristic', heuristic)):
            dm, tn, mr, de = run(f)
            print('%-28s %9.1f %8.1f %11.1f%% %9.0f%%' % (name, dm, tn, mr, de))
        print('\nA trained agent is only interesting once its match rate is well '
              'above 10% and\nits turns-to-win drops below 26 -- that is where '
              'strategy starts to matter.')


if __name__ == '__main__':
    main()
