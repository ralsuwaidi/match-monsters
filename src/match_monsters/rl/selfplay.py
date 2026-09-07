"""Two-sided game state for neural self-play. No torch, no heuristics.

Neither agent is told what a tile means, what mana is for, or which colours
belong to it. Each sees the raw board, its own and the opponent's bars, and the
set of legal actions. Everything else has to come from winning and losing.

A `Duel` steps one decision at a time for whichever side is on move, so both
agents generate trajectories from the same game.
"""
import random
from collections import Counter

import numpy as np

from match_monsters.game import engine
from match_monsters.game import grid
from match_monsters import rules

TILES = grid.COLORS + [grid.BERRY]
TILE_IX = {t: i for i, t in enumerate(TILES)}

# 6 tile planes + 2 marking the colours that feed MY monsters + 2 for THEIRS.
# The ownership planes are not a heuristic -- the game shows you which element
# each monster consumes. Giving the net that mapping is what lets a SINGLE
# network play either team instead of memorising one.
N_PLANES = len(TILES) + 4
MON_FEATURES = 11
SCALARS = MON_FEATURES * 4 + 6

N_H = grid.H * (grid.W - 1)
N_V = (grid.H - 1) * grid.W
N_SWAP = N_H + N_V
N_ACTIONS = N_SWAP + 2


def _swaps():
    out = []
    for r in range(grid.H):
        for c in range(grid.W - 1):
            out.append((r, c, r, c + 1))
    for r in range(grid.H - 1):
        for c in range(grid.W):
            out.append((r, c, r + 1, c))
    return out


SWAPS = _swaps()
assert len(SWAPS) == N_SWAP
SWAP_IX = {s: i for i, s in enumerate(SWAPS)}


class Duel:
    """One game. `active` is 0 for the first team, 1 for the second."""

    def __init__(self, rng, teams, max_turns=60):
        self.py_rng = rng
        self.g = grid.Grid(rng)
        self.teams = teams
        hp = [rules.BASE_HP, rules.BASE_HP + rules.SECOND_PLAYER_HP_BONUS]
        self.sides = [engine.Side(teams[i], hp[i], 'me' if i == 0 else 'foe')
                      for i in (0, 1)]
        self.st = Counter()
        self.active = 0
        self.moves = rules.MOVES_PER_TURN
        self.extras = 0
        self.turn = 0
        self.max_turns = max_turns
        self.done = False
        self.result = None            # 1.0 = side 0 won, 0.0 = side 1 won
        # skill telemetry: a random agent wastes most of its moves
        self.n_swaps = 0
        self.n_matched = 0
        self.n_big = 0                # matches of 4+, which refund the move
        self.last_matched = False     # did the most recent action clear tiles?

    def clone(self):
        """Independent copy, for search. The RNG is forked so each rollout
        samples its own refill -- the board after a match is genuinely random,
        so a single lookahead is one sample of many."""
        import copy
        d = object.__new__(Duel)
        d.py_rng = random.Random(self.py_rng.randrange(1 << 30))
        d.g = self.g.clone(rng=d.py_rng)
        d.teams = self.teams
        d.sides = [copy.deepcopy(s) for s in self.sides]
        d.st = Counter()
        d.active = self.active
        d.moves = self.moves
        d.extras = self.extras
        d.turn = self.turn
        d.max_turns = self.max_turns
        d.done = self.done
        d.result = self.result
        d.n_swaps = d.n_matched = d.n_big = 0
        d.last_matched = False
        return d

    # ---------------------------------------------------------------- obs --
    def legal_mask(self, match_only=False):
        """`match_only` is a CURRICULUM, not a rule: it hides the swaps that
        make no match. Random search never finds the chain
        swap -> match -> mana -> threshold -> strike -> damage on its own, so
        early training restricts the agent to moves that do something, and the
        restriction is annealed away. Which match to make, when to fire, when
        to evolve and when to reposition are all still learned."""
        m = np.zeros(N_ACTIONS, np.int8)
        g = self.g.g
        for i, (a, b, c, d) in enumerate(SWAPS):
            if g[a][b] != g[c][d]:
                m[i] = 1
        if match_only:
            mm = np.zeros(N_ACTIONS, np.int8)
            for mv, _cl, _ln in self.g.all_swaps(None, want_setups=False)[0]:
                mm[SWAP_IX[mv]] = 1
            if mm.any():
                m = mm
        side = self.sides[self.active]
        if side.berries >= rules.BERRIES_TO_EVOLVE:
            for j, colour in enumerate(side.mons):
                if not side.evolved[colour] or rules.ALLOW_BOOST:
                    m[N_SWAP + j] = 1
        if not m.any():
            m[0] = 1
        return m

    def observe(self):
        """Side-relative. Everything is described from the point of view of the
        side on move, and every monster is described by its STATS rather than
        its identity -- so one network can play either team and generalise."""
        board = np.zeros((N_PLANES, grid.H, grid.W), np.float32)
        gg = self.g.g
        for r in range(grid.H):
            row = gg[r]
            for c in range(grid.W):
                board[TILE_IX[row[c]], r, c] = 1.0
        me = self.sides[self.active]
        foe = self.sides[1 - self.active]
        for k, colour in enumerate(me.mons):
            board[len(TILES) + k] = board[TILE_IX[colour]]
        for k, colour in enumerate(foe.mons):
            board[len(TILES) + 2 + k] = board[TILE_IX[colour]]

        s = np.zeros(SCALARS, np.float32)
        i = 0
        for side in (me, foe):
            for colour, mon in side.mons.items():
                ev = side.evolved[colour]
                ab = mon.ability(ev, side.charges[colour])
                top = mon.ability(ev, mon.cap(ev))
                s[i] = side.mana[colour] / mon.cost; i += 1
                s[i] = mon.cost / 10.0; i += 1
                s[i] = ab.damage / 25.0; i += 1
                s[i] = top.damage / 25.0; i += 1
                s[i] = 1.0 if ev else 0.0; i += 1
                s[i] = 1.0 if mon.charged else 0.0; i += 1
                s[i] = (side.charges[colour] / 3.0) if mon.charged else 0.0; i += 1
                s[i] = mon.cap(ev) / 3.0; i += 1
                s[i] = ab.drain / 3.0; i += 1
                s[i] = (ab.convert_n + ab.collect_n) / 8.0; i += 1
                s[i] = ab.clear_columns / 2.0; i += 1
        for side in (me, foe):
            s[i] = side.berries / rules.BERRY_CAP; i += 1
            s[i] = side.hp / rules.BASE_HP; i += 1
        s[i] = self.moves / 3.0; i += 1
        s[i] = float(self.extras); i += 1
        return board, s

    # --------------------------------------------------------------- step --
    def step(self, action):
        """Apply one decision for the side on move.

        Returns True when the side's turn has ended (control passes over)."""
        me = self.sides[self.active]
        foe = self.sides[1 - self.active]
        action = int(action)

        self.last_matched = False
        if action >= N_SWAP:
            colour = list(me.mons)[action - N_SWAP]
            if me.berries >= rules.BERRIES_TO_EVOLVE:
                me.berries -= rules.BERRIES_TO_EVOLVE
                if not me.evolved[colour]:
                    me.evolved[colour] = True
                    # record it the same way engine.play_turn does, or the
                    # telemetry reports zero evolutions for everyone
                    self.st[me.label + '/evolve_' + me.mons[colour].name] += 1
                else:
                    me.mana[colour] += rules.BOOST_MANA
                    self.st[me.label + '/boosts'] += 1
                engine.fire_all(me, foe, self.g, self.st)
            self.moves -= 1
        else:
            a, b, c, d = SWAPS[action]
            self.moves -= 1
            self.n_swaps += 1
            if self.g.g[a][b] != self.g.g[c][d]:
                self.g.g[a][b], self.g.g[c][d] = self.g.g[c][d], self.g.g[a][b]
                hits, _ = self.g._runs_near(((a, b), (c, d)))
                if hits:
                    self.n_matched += 1
                    self.last_matched = True
                    cleared, longest = self.g.resolve()
                    if longest >= 4:
                        self.n_big += 1
                    engine.collect(me, cleared, self.st, foe)
                    if longest >= 4 and self.extras < rules.EXTRA_MOVES_PER_TURN:
                        self.extras += 1
                        self.moves += 1
                    me.ability_longest = 0
                    engine.fire_all(me, foe, self.g, self.st)
                    if (rules.EXTRA_MOVE_FROM_ABILITY and me.ability_longest >= 4
                            and self.extras < rules.EXTRA_MOVES_PER_TURN):
                        self.extras += 1
                        self.moves += 1

        if foe.hp <= 0:
            self.done = True
            self.result = 1.0 if self.active == 0 else 0.0
            return True

        if self.moves <= 0:
            for cc, mon in me.mons.items():
                if mon.charges_per_turn:
                    me.charges[cc] = min(me.charges[cc] + mon.charges_per_turn,
                                         mon.cap(me.evolved[cc]))
            self.turn += 1
            if self.turn >= self.max_turns:
                # A timeout is not a real draw. Grade it by HP so that being
                # ahead when the clock runs out is still worth something --
                # otherwise two random agents produce nothing but 0.5s and
                # there is no gradient to learn from.
                self.done = True
                d = (self.sides[0].hp - self.sides[1].hp) / rules.BASE_HP
                self.result = 0.5 + 0.5 * max(-1.0, min(1.0, d))
                return True
            self.active = 1 - self.active
            nxt = self.sides[self.active]
            self.moves = rules.MOVES_PER_TURN + nxt.extra_moves_pending
            nxt.extra_moves_pending = 0
            self.extras = 0
            return True
        return False
