"""How a side decides what to play.

Moves are scored in damage-equivalent units: each monster's per-activation
value is solved as a fixed point (abilities feed each other), a tile is worth
that divided by the mana cost, and mana is valued against the firing threshold
rather than linearly.
"""

from dataclasses import dataclass

from match_monsters.game import grid
from match_monsters import rules


# ----------------------------------------------------------- VALUATION ----
@dataclass
class Ctx:
    worth: dict          # my colour -> value of one tile of it
    foe_worth: dict      # enemy colour -> value of one tile of it (denial)
    berry: float         # value of one berry
    move_value: float    # value of an average move
    act: dict            # (colour, charges) -> value of one activation
    rank: dict           # tile -> weight, for ranking setup swaps
    ub: float = 0.0      # upper bound on any single move's score


def _ability_value(ab, w_self, w_foe, n_foe_mons, move_value, berry, hp_cap):
    # damage past the opponent's remaining HP is wasted, so cap it
    v = float(min(ab.damage, hp_cap)) + ab.heal
    if ab.convert_n and ab.convert_color:
        v += ab.convert_n * w_self.get(ab.convert_color, 0.0)
    cells = ab.clear_columns * grid.H + ab.clear_rows * grid.W + ab.collect_n
    if cells:
        # a cleared line holds, on average, cells/6 of each of the 6 tile types
        v += cells / 6.0 * (sum(w_self.values()) + berry)
    if ab.drain and w_foe:
        v += ab.drain * n_foe_mons * (sum(w_foe.values()) / len(w_foe))
    v += ab.extra_moves * move_value
    return v


def _nominal(side, foe, w_self, w_foe, mv, berry, hp_cap):
    out = {}
    for c, m in side.mons.items():
        ab = m.ability(side.evolved[c], m.cap(side.evolved[c]))
        out[c] = _ability_value(ab, w_self, w_foe, len(foe.mons), mv, berry,
                                hp_cap) / m.cost
    return out


def valuation(side, foe, pol=None):
    """Fixed-point estimate of what one tile of each colour is worth, in
    damage-equivalent units. Converges because abilities feed each other."""
    hp_s, hp_f = max(1, foe.hp), max(1, side.hp)
    ws = {c: min(side.mons[c].ability(side.evolved[c],
                                      side.mons[c].cap(side.evolved[c])).damage,
                 hp_s) / side.mons[c].cost for c in side.mons}
    wf = {c: min(foe.mons[c].ability(foe.evolved[c],
                                     foe.mons[c].cap(foe.evolved[c])).damage,
                 hp_f) / foe.mons[c].cost for c in foe.mons}
    berry_s = berry_f = 0.0
    mv_s = mv_f = 0.0

    for _ in range(6):
        mv_s = rules.AVG_TILES_PER_MOVE * (sum(ws.values()) / len(ws))
        mv_f = rules.AVG_TILES_PER_MOVE * (sum(wf.values()) / len(wf))
        ws, wf = (_nominal(side, foe, ws, wf, mv_s, berry_s, hp_s),
                  _nominal(foe, side, wf, ws, mv_f, berry_f, hp_f))
        berry_s = _berry_worth(side, foe, ws, wf, mv_s, hp_s)
        berry_f = _berry_worth(foe, side, wf, ws, mv_f, hp_f)

    act = {}
    for c, m in side.mons.items():
        for ch in range(m.cap(side.evolved[c]) + 1):
            act[(c, ch)] = _ability_value(m.ability(side.evolved[c], ch), ws, wf,
                                          len(foe.mons), mv_s, berry_s, hp_s)

    own_w = pol.own_w if pol else 1.0
    berry_w = pol.berry_w if pol else 1.0
    deny_w = pol.deny_w if pol else 0.0
    neutral_w = pol.neutral_w if pol else 0.0
    extra_w = pol.extra_w if pol else 1.0

    rank = dict(ws)
    rank[grid.BERRY] = berry_s * berry_w
    for c, w in wf.items():
        rank[c] = w * deny_w

    # Loose but valid ceiling on what any one move can score. A swap clears at
    # most 20 tiles (two runs through each of two adjacent cells) and can set
    # off at most two activations. Used to skip the setup search outright when
    # no follow-up could possibly clear the threshold.
    per_tile = max(max(ws.values()) * own_w, berry_s * berry_w,
                   (max(wf.values()) * deny_w) if wf else 0.0, neutral_w, 0.0)
    max_act = (max(act.values()) if act else 0.0) * (1.0 + rules.TEMPO_BONUS)
    ub = 20.0 * per_tile + 2.0 * max_act + extra_w * mv_s

    return Ctx(worth=ws, foe_worth=wf, berry=berry_s, move_value=mv_s,
               act=act, rank=rank, ub=ub)


def _berry_worth(s, f, w_self, w_other, mv, hp_cap):
    """A berry is 1/4 of an evolution. Worth the move it costs to cash in?"""
    best = 0.0
    for c, m in s.mons.items():
        if s.evolved[c]:
            gain = rules.BOOST_MANA * w_self[c]
        else:
            b = _ability_value(m.ability(False, m.cap(False)), w_self, w_other,
                               len(f.mons), mv, 0.0, hp_cap)
            e = _ability_value(m.ability(True, m.cap(True)), w_self, w_other,
                               len(f.mons), mv, 0.0, hp_cap)
            gain = (e - b) * rules.EXPECTED_ACTIVATIONS_AFTER_EVO
        best = max(best, gain)
    net = best - (mv if rules.EVOLVE_COSTS_MOVE else 0.0)
    return max(0.0, net) / rules.BERRIES_TO_EVOLVE


def mana_gain_value(side, ctx, color, n):
    """Value of gaining n mana on `color`, aware of thresholds and charges.

    Firing Yellow at 0 charges does nothing, so pushing its bar over the line
    at the wrong moment scores NEGATIVE -- the model knows mana was thrown away."""
    mon = side.mons[color]
    k, m, ch = mon.cost, side.mana[color], side.charges[color]
    if not rules.MANA_CARRYOVER:
        # the bar zeroes on firing, so every point past the threshold is burnt;
        # one collection can therefore only ever trigger a single strike
        if m + n >= k:
            return ctx.act.get((color, ch), 0.0) * (1.0 + rules.TEMPO_BONUS)
        return n * ctx.worth[color]
    fires = (m + n) // k - m // k
    val = 0.0
    for i in range(fires):
        val += ctx.act.get((color, ch if i == 0 else 0), 0.0) * (1.0 + rules.TEMPO_BONUS)
        ch = 0
    return val + (n - fires * k) * ctx.worth[color]


# -------------------------------------------------------------- POLICY ----
@dataclass(frozen=True)
class Policy:
    name: str
    own_w: float = 1.0
    deny_w: float = 0.0
    berry_w: float = 1.0
    neutral_w: float = 0.0
    extra_w: float = 1.0
    use_berries: bool = True
    evolve_target: str = ''
    # Evolve these in order, then boost. Empty means the single evolve_target
    # forever (evolve it, then spend every later set of 4 berries boosting it).
    evolve_order: tuple = ()
    # 'score'      -- weigh everything together (the original policies)
    # 'self_first' -- strict ladder: help yourself, and only ruin the opponent
    #                 with a move that could not have helped you
    mode: str = 'score'
    ruin_cap: int = 1           # most ruin MATCHES allowed in one turn
    # How much of the upside you forgo by firing a charge monster early is
    # counted against the move. 0 = fire whenever the bar fills; 1 = the full
    # difference between this tier and the top tier is charged against it.
    # Because the penalty scales with what you give up, it produces graded
    # behaviour: never at honey 0, hold at 1, situational at 2, always at cap.
    honey_patience: float = 0.0
    setup: bool = True          # will it spend a move repositioning a tile?
    # How dearly a repositioning move is priced. Measured: at 1.0 both sides
    # over-use it and play worse; 2.0+ is where each plays its best, and
    # repositioning settles at ~2-4% of moves -- a rescue, not a strategy.
    setup_cost_w: float = 2.0


NICE_COLOUR = {'red': 'Bonzumi', 'yellow': 'Sipzap',
               'blue': 'Pelijet', 'purple': 'Barbenin'}


def describe(pol):
    """Plain-English account of what a policy actually does."""
    who = lambda c: NICE_COLOUR.get(c, c)
    bits = []

    if pol.mode == 'self_first':
        bits.append(
            "**Strict ladder.** Always takes the match that gains it the most "
            "mana or berries. Only when *no* match gains it anything does it "
            "spend the move attacking the opponent's tiles"
            + (f", and never more than {pol.ruin_cap} such move"
               f"{'s' if pol.ruin_cap != 1 else ''} in a turn."
               if pol.ruin_cap else " -- and here, never at all."))
    else:
        d = pol.deny_w
        level = ("ignores the opponent's tiles entirely" if d == 0 else
                 "gives the opponent's tiles modest weight" if d <= 0.6 else
                 "weights the opponent's tiles heavily, taking denial matches "
                 "over weak matches of its own")
        bits.append(f"**Weighted scoring.** Scores every legal match in "
                    f"damage-equivalent units and takes the best; it {level} "
                    f"(denial weight {d:g}).")

    if not pol.use_berries:
        bits.append("**Never evolves** -- refuses to spend a move cashing berries in.")
    elif pol.evolve_order:
        seq = " then ".join(who(c) for c in pol.evolve_order)
        bits.append(f"**Evolves {seq}**, then spends later berries on boosts.")
    else:
        t = who(pol.evolve_target)
        bits.append(f"**Evolves {t}**, then spends every later set of 4 berries "
                    f"boosting {t} by 4 mana rather than evolving the other one.")

    if pol.berry_w != 1.0:
        bits.append(f"Values berries {pol.berry_w:g}x the normal amount.")

    if not pol.setup:
        bits.append("Never repositions a tile deliberately.")
    elif pol.mode != 'self_first':
        bits.append(
            f"Will spend a move repositioning a tile only if the match that "
            f"unlocks beats the best match available now, plus "
            f"{pol.setup_cost_w:g}x the value of an ordinary move.")
    return bits


def score_move(side, foe, ctx, pol, cleared, longest, bonus_avail):
    s = 0.0
    for t, n in cleared.items():
        if t == grid.BERRY:
            s += pol.berry_w * ctx.berry * n
        elif t in side.mons:
            v = pol.own_w * mana_gain_value(side, ctx, t, n)
            mon = side.mons[t]
            if (pol.honey_patience and mon.charged
                    and side.mana[t] + n >= mon.cost):
                cap = mon.cap(side.evolved[t])
                h = side.charges[t]
                if h < cap:
                    forgone = (ctx.act.get((t, cap), 0.0)
                               - ctx.act.get((t, h), 0.0))
                    v -= pol.honey_patience * forgone
            s += v
        elif t in foe.mons:
            s += pol.deny_w * ctx.foe_worth[t] * n
        else:
            s += pol.neutral_w * n
    if longest >= 4 and bonus_avail:
        s += pol.extra_w * ctx.move_value
    return s


def self_value(side, ctx, pol, cleared, longest, bonus_avail):
    """What this match is worth to ME -- mana and berries only."""
    v = 0.0
    for t, n in cleared.items():
        if t == grid.BERRY:
            v += pol.berry_w * ctx.berry * n
        elif t in side.mons:
            v += pol.own_w * mana_gain_value(side, ctx, t, n)
    if longest >= 4 and bonus_avail:
        v += pol.extra_w * ctx.move_value
    return v


def foe_value(foe, ctx, cleared):
    """What this match takes away from THEM."""
    return sum(n * ctx.foe_worth.get(t, 0.0)
               for t, n in cleared.items() if t in foe.mons)


def foe_best_available(g, foe, ctx):
    """How good the board currently looks for the opponent."""
    best = 0.0
    for _mv, cleared, longest in g.moves_with_preview():
        v = foe_value(foe, ctx, cleared)
        if longest >= 4:
            v += ctx.move_value
        if v > best:
            best = v
    return best


def choose_move_self_first(side, foe, g, ctx, pol, matches, setups,
                           bonus_avail, moves_left, ruin_used):
    """Strict ladder, in order:

    1. the match that gains me the most mana or berries;
    2. if no match gains me anything -- ruin them by matching their colours,
       but never more than `ruin_cap` times in a turn;
    3. if there is still nothing, reposition: set myself up while I have a move
       left to use it, otherwise stack a tile to spoil their board.
    """
    best_mv, best_sc = None, 1e-9
    for mv, cleared, longest in matches:
        sc = self_value(side, ctx, pol, cleared, longest, bonus_avail)
        if sc > best_sc:
            best_sc, best_mv = sc, mv
    if best_mv is not None:
        return ('match', best_mv, False)

    if ruin_used < pol.ruin_cap:
        rm, rsc = None, 1e-9
        for mv, cleared, longest in matches:
            sc = foe_value(foe, ctx, cleared)
            if sc > rsc:
                rsc, rm = sc, mv
        if rm is not None:
            return ('match', rm, True)

    if setups:
        if moves_left >= 2:
            # a move is still coming, so open something for myself
            sm, sv = None, 1e-9
            for mv, _pot in setups:
                clone = g.clone(rng=g.rng)
                clone.swap(mv)
                for _m2, cleared, longest in clone.moves_with_preview_near(
                        ((mv[0], mv[1]), (mv[2], mv[3]))):
                    v = self_value(side, ctx, pol, cleared, longest, bonus_avail)
                    if v > sv:
                        sv, sm = v, mv
            if sm is not None:
                return ('setup', sm, False)
        # last move and nothing for me: leave the board as bad as possible
        # for them. Rare (~2% of slots), so an exact search is affordable.
        before = foe_best_available(g, foe, ctx)
        worst, worst_v = None, before
        for mv, _pot in setups:
            clone = g.clone(rng=g.rng)
            clone.swap(mv)
            v = foe_best_available(clone, foe, ctx)
            if v < worst_v:
                worst_v, worst = v, mv
        if worst is not None:
            return ('setup', worst, False)
        return ('setup', setups[0][0], False)

    if matches:
        return ('match', matches[0][0], False)
    return (None, None, False)


def choose_move(side, foe, g, ctx, pol, matches, setups, bonus_avail, moves_left):
    """Pick a match, or spend this move repositioning a tile to set up a bigger
    one next move. A setup is taken only if the match it unlocks beats the best
    match available now PLUS the value of the move it costs."""
    best_sc, best_mv = None, None
    for mv, cleared, longest in matches:
        sc = score_move(side, foe, ctx, pol, cleared, longest, bonus_avail)
        if best_sc is None or sc > best_sc:
            best_sc, best_mv = sc, mv

    if best_mv is None:
        # nothing is matchable: find the reposition that opens the best match
        best_setup, best_val = None, None
        for mv, _pot in setups:
            clone = g.clone(rng=g.rng)
            clone.swap(mv)
            for mv2, cleared, longest in clone.moves_with_preview():
                sc2 = score_move(side, foe, ctx, pol, cleared, longest, bonus_avail)
                if best_val is None or sc2 > best_val:
                    best_val, best_setup = sc2, mv
        return ('setup', best_setup, False) if best_setup else (None, None, False)

    can_setup = (rules.ALLOW_NON_MATCHING_SWAP and pol.setup and setups)
    if not can_setup or moves_left < 2:
        return ('match', best_mv, False)

    # a setup costs you this move, so the match it unlocks must beat the best
    # match available now PLUS what an ordinary next move would have been worth
    threshold = best_sc + pol.setup_cost_w * max(best_sc, ctx.move_value)
    if ctx.ub <= threshold:
        return ('match', best_mv, False)   # nothing a setup unlocks could clear it

    best_setup, best_val = None, threshold
    for mv, _pot in setups[:rules.SETUP_SEARCH_K]:
        clone = g.clone(rng=g.rng)     # no refill happens, so sharing RNG is safe
        clone.swap(mv)
        # only matches the setup could have changed can beat the threshold
        for mv2, cleared, longest in clone.moves_with_preview_near(
                ((mv[0], mv[1]), (mv[2], mv[3]))):
            sc2 = score_move(side, foe, ctx, pol, cleared, longest, bonus_avail)
            if sc2 > best_val:
                best_val, best_setup = sc2, mv
    if best_setup is not None:
        return ('setup', best_setup, False)
    return ('match', best_mv, False)



# ------------------------------------------------------------ POLICIES ----
def _mk(name, **kw):
    return Policy(name=name, **kw)

MY_POLICIES = {p.name: p for p in [
    _mk('bon_evo',     deny_w=0.0, berry_w=1.0, evolve_target='red'),
    _mk('sip_evo',     deny_w=0.0, berry_w=1.0, evolve_target='yellow'),
    _mk('bon_deny',    deny_w=0.6, berry_w=1.0, evolve_target='red'),
    _mk('sip_deny',    deny_w=0.6, berry_w=1.0, evolve_target='yellow'),
    _mk('bon_hdeny',   deny_w=1.2, berry_w=1.0, evolve_target='red'),
    _mk('bon_then_sip', deny_w=1.2, berry_w=1.0, evolve_order=('red', 'yellow')),
    _mk('sip_then_bon', deny_w=1.2, berry_w=1.0, evolve_order=('yellow', 'red')),
    # one-turn probes used by rollout.py
    _mk('line_build', deny_w=0.0, berry_w=1.0, evolve_target='red'),
    _mk('line_deny',  deny_w=3.0, berry_w=1.0, evolve_target='red'),
    # from the played game: hold yellow for the honey-3 tier
    _mk('sip_hold',   deny_w=1.2, berry_w=1.0, evolve_target='yellow',
        honey_patience=1.0),
    _mk('bon_hold',   deny_w=1.2, berry_w=1.0, evolve_target='red',
        honey_patience=1.0),
    _mk('bon_hp25',   deny_w=1.2, berry_w=1.0, evolve_target='red',
        honey_patience=0.25),
    _mk('sip_hp25',   deny_w=1.2, berry_w=1.0, evolve_target='yellow',
        honey_patience=0.25),
    _mk('bon_hp50',   deny_w=1.2, berry_w=1.0, evolve_target='red',
        honey_patience=0.50),
    _mk('sip_hp50',   deny_w=1.2, berry_w=1.0, evolve_target='yellow',
        honey_patience=0.50),
    _mk('bon_hp75',   deny_w=1.2, berry_w=1.0, evolve_target='red',
        honey_patience=0.75),
    _mk('sip_hp75',   deny_w=1.2, berry_w=1.0, evolve_target='yellow',
        honey_patience=0.75),
    _mk('bon_hp125',   deny_w=1.2, berry_w=1.0, evolve_target='red',
        honey_patience=1.25),
    _mk('sip_hp125',   deny_w=1.2, berry_w=1.0, evolve_target='yellow',
        honey_patience=1.25),
    # the strict ladder: self first, ruin only with a move that was no use to me
    _mk('self_first',    mode='self_first', ruin_cap=1, evolve_target='red'),
    _mk('self_first_seq', mode='self_first', ruin_cap=1,
        evolve_order=('red', 'yellow')),
    _mk('self_never_ruin', mode='self_first', ruin_cap=0, evolve_target='red'),
    _mk('self_ruin_twice', mode='self_first', ruin_cap=2, evolve_target='red'),
    _mk('sip_hdeny',   deny_w=1.2, berry_w=1.0, evolve_target='yellow'),
    _mk('sip_berry2',  deny_w=0.6, berry_w=2.0, evolve_target='yellow'),
    _mk('bon_berry2',  deny_w=0.6, berry_w=2.0, evolve_target='red'),
    _mk('no_berry',    deny_w=0.0, berry_w=0.0, use_berries=False, evolve_target='red'),
    # AI-strength variants: identical to bon_hdeny except for the setup rule,
    # so the comparison isolates repositioning and nothing else
    _mk('bon_nosetup', deny_w=1.2, berry_w=1.0, evolve_target='red', setup=False),
    _mk('bon_setup05', deny_w=1.2, berry_w=1.0, evolve_target='red', setup_cost_w=0.5),
    _mk('bon_setup15', deny_w=1.2, berry_w=1.0, evolve_target='red', setup_cost_w=1.5),
    _mk('bon_setup20', deny_w=1.2, berry_w=1.0, evolve_target='red', setup_cost_w=2.0),
    _mk('bon_setup30', deny_w=1.2, berry_w=1.0, evolve_target='red', setup_cost_w=3.0),
]}
STRATEGIES_ME = ['bon_evo', 'bon_deny', 'bon_hdeny', 'bon_berry2',
                 'bon_then_sip', 'sip_then_bon',
                 'self_first', 'self_first_seq', 'self_ruin_twice',
                 'sip_evo', 'sip_deny', 'sip_hdeny', 'sip_berry2', 'no_berry']

FOE_POLICIES = {p.name: p for p in [
    _mk('pel_evo',     deny_w=0.0, berry_w=1.0, evolve_target='blue'),
    _mk('bar_evo',     deny_w=0.0, berry_w=1.0, evolve_target='purple'),
    _mk('pel_deny',    deny_w=0.6, berry_w=1.0, evolve_target='blue'),
    _mk('pel_then_bar', deny_w=0.6, berry_w=1.0, evolve_order=('blue', 'purple')),
    _mk('bar_then_pel', deny_w=0.6, berry_w=1.0, evolve_order=('purple', 'blue')),
    _mk('self_first',    mode='self_first', ruin_cap=1, evolve_target='blue'),
    _mk('self_first_seq', mode='self_first', ruin_cap=1,
        evolve_order=('blue', 'purple')),
    _mk('bar_deny',    deny_w=0.6, berry_w=1.0, evolve_target='purple'),
    _mk('pel_hdeny',   deny_w=1.2, berry_w=1.0, evolve_target='blue'),
    _mk('no_berry',    deny_w=0.0, berry_w=0.0, use_berries=False, evolve_target='blue'),
    _mk('pel_nosetup', deny_w=0.6, berry_w=1.0, evolve_target='blue', setup=False),
    _mk('pel_setup20', deny_w=0.6, berry_w=1.0, evolve_target='blue', setup_cost_w=2.0),
]}
STRATEGIES_FOE = ['pel_evo', 'bar_evo', 'pel_deny', 'bar_deny', 'pel_hdeny',
                  'pel_then_bar', 'bar_then_pel',
                  'self_first', 'self_first_seq', 'no_berry']

