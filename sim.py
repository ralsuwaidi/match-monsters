"""Match Monsters duel simulator -- facade over the split modules.

    python3 sim.py            # everything
    python3 sim.py board | matrix | head | sens | ai

The real code lives in:
    rules.py    every rule constant, [C]onfirmed or [A]ssumed
    grid.py     the board
    monsters.py monster stats and abilities
    ai.py       valuation, policies, move choice
    engine.py   sides, mana, firing, turns, a duel
    runner.py   multiprocessing and confidence intervals
    reports.py  the printed reports
"""

from rules import *                                              # noqa: F401,F403
import rules                                                     # noqa: F401
import grid                                                      # noqa: F401
import monsters                                                  # noqa: F401
from monsters import MY_TEAM, FOE_TEAM                            # noqa: F401
from ai import (Ctx, Policy, valuation, mana_gain_value,          # noqa: F401
                score_move, choose_move, MY_POLICIES, FOE_POLICIES,
                STRATEGIES_ME, STRATEGIES_FOE)
import engine                                                    # noqa: F401
from engine import Side, collect, fire, fire_all, play_turn, duel  # noqa: F401
from runner import apply_cfg, run, wilson                        # noqa: F401
import reports                                                   # noqa: F401
from reports import (report_board, report_matrix, report_head,    # noqa: F401
                     report_sens, report_ai)

if __name__ == '__main__':
    reports.main()
