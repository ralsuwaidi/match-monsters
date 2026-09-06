"""Every rule constant in one place.

[C] confirmed by the official Plato rules page or by the user in game
[A] assumption -- each one is measured in reports.report_sens

Modules read these as `rules.NAME` rather than importing the values, so the
sensitivity sweeps can change one at run time and have it take effect.
"""

# [C] confirmed by the official Plato rules page or by the user
# [A] assumption -- swept in `sens`

# [C] Official base HP. The 70/75 seen in the sample screenshots was a
# temporary weekly event, not the standard rule.
BASE_HP = 80
SECOND_PLAYER_HP_BONUS = 5      # [C] official: player 2 gets +5 HP
MOVES_PER_TURN = 2              # [C]
EXTRA_MOVES_PER_TURN = 1        # [C] Normal mode: a 4+ match gives ONE extra move
# [C] A 4+ match formed by an ability -- Pelijet's conversion landing beside two
# blues, or a cascade after Flare clears a column -- also earns the extra move,
# subject to the same once-per-turn cap.
EXTRA_MOVE_FROM_ABILITY = True
BERRIES_TO_EVOLVE = 4           # [C]
BERRY_CAP = 4                   # [C] the counter holds at most 4; extras are wasted
EVOLVE_COSTS_MOVE = True        # [C] official: evolving consumes a move slot
ALLOW_NON_MATCHING_SWAP = True  # [C] you may spend a move repositioning a tile
ALLOW_BOOST = True              # [C] evolved monster + 4 berries -> mana boost
BOOST_MANA = 4                  # [C] 4 mana, to the chosen monster only
MANA_CARRYOVER = False          # [C] a full bar zeroes out; overflow is LOST
AUTO_FIRE = True                # [C] a full mana bar fires immediately
BEST_LINE = False               # [C] Flare matches a RANDOM column, not a chosen one
MAX_TURNS = 200

# --- AI tuning (affects how well the sim plays, not the rules) ---
SETUP_SEARCH_K = 6              # how many setup swaps get searched properly
AVG_TILES_PER_MOVE = 3.4
TEMPO_BONUS = 0.15
EXPECTED_ACTIVATIONS_AFTER_EVO = 4.0

