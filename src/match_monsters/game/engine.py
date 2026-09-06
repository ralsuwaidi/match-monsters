"""The game itself: sides, mana, abilities, turns, and a full duel."""

from collections import Counter

from match_monsters.game import grid
from match_monsters.game.grid import Grid
from match_monsters import rules
from match_monsters.agents.ai import valuation, choose_move, choose_move_self_first
from match_monsters.game.monsters import MY_TEAM, FOE_TEAM


TRACE = None            # set to a callable to get a play-by-play


def _log(kind, **kw):
    if TRACE is not None:
        TRACE(kind, **kw)


# ----------------------------------------------------------------- SIDE ----
class Side:
    def __init__(self, team, hp, label):
        self.label = label
        self.mons = {m.color: m for m in team}
        self.mana = {c: 0 for c in self.mons}
        self.evolved = {c: False for c in self.mons}
        self.charges = {c: 0 for c in self.mons}
        self.berries = 0
        self.hp = hp
        self.max_hp = hp
        self.extra_moves_pending = 0
        self.ability_longest = 0   # longest run an ability made this turn


def _merge(dst, src):
    for t, n in src.items():
        dst[t] = dst.get(t, 0) + n
    return dst


def collect(side, cleared, st):
    for t, n in cleared.items():
        if t == grid.BERRY:
            got = min(n, rules.BERRY_CAP - side.berries)
            side.berries += got
            st[side.label + '/berries'] += got
            st[side.label + '/berries_wasted'] += n - got
        elif t in side.mana:
            side.mana[t] += n
            st[side.label + '/mana_' + t] += n


# -------------------------------------------------------------- FIRING ----
def fire(side, foe, g, color, st):
    mon = side.mons[color]
    charges_before = side.charges[color]
    ab = mon.ability(side.evolved[color], charges_before)

    if rules.MANA_CARRYOVER:
        side.mana[color] -= mon.cost
    else:
        # the bar zeroes out, so anything above the threshold is burnt
        st[side.label + '/wasted_' + mon.name] += side.mana[color] - mon.cost
        side.mana[color] = 0
    st[side.label + '/spent_' + mon.name] += mon.cost
    if mon.charged:
        side.charges[color] = 0

    st[side.label + '/fires_' + mon.name] += 1
    if ab.damage:
        foe.hp -= ab.damage
        st[side.label + '/dmg_' + mon.name] += ab.damage
    if ab.heal:
        side.hp = min(side.max_hp, side.hp + ab.heal)
    if ab.drain:
        for c2 in foe.mons:
            foe.mana[c2] = max(0, foe.mana[c2] - ab.drain)
        st[side.label + '/drained'] += ab.drain * len(foe.mons)
    if ab.extra_moves:
        side.extra_moves_pending += ab.extra_moves

    gained = {}
    longest_from_ability = 0
    own = tuple(side.mons) + (grid.BERRY,)
    for _ in range(ab.clear_columns):
        c = g.best_column(own) if rules.BEST_LINE else g.rng.randrange(grid.W)
        _merge(gained, g.clear_column(c))
        longest_from_ability = max(longest_from_ability, g.last_longest)
    for _ in range(ab.clear_rows):
        r = g.best_row(own) if rules.BEST_LINE else g.rng.randrange(grid.H)
        _merge(gained, g.clear_row(r))
        longest_from_ability = max(longest_from_ability, g.last_longest)
    if ab.convert_n and ab.convert_color:
        _merge(gained, g.convert(ab.convert_n, ab.convert_color))
        longest_from_ability = max(longest_from_ability, g.last_longest)
    if ab.collect_n:
        _merge(gained, g.collect_random(ab.collect_n))
        longest_from_ability = max(longest_from_ability, g.last_longest)
    side.ability_longest = max(getattr(side, 'ability_longest', 0),
                               longest_from_ability)
    collect(side, gained, st)
    _log('fire', side=side, foe=foe, g=g, mon=mon, ab=ab,
         charges=charges_before, gained=gained)


def fire_all(side, foe, g, st):
    """Fire every full bar, repeating if an ability's tiles refill one."""
    for _ in range(10):
        fired = False
        for color, mon in side.mons.items():
            if side.mana[color] >= mon.cost:
                fire(side, foe, g, color, st)
                fired = True
                if foe.hp <= 0:
                    return
        if not fired:
            return



# ---------------------------------------------------------------- TURN ----
def play_turn(side, foe, g, pol, st):
    _log('turn_start', side=side, foe=foe, g=g)
    ctx = valuation(side, foe, pol)
    moves = rules.MOVES_PER_TURN + side.extra_moves_pending
    side.extra_moves_pending = 0
    extras = 0
    ruin_used = 0

    for _guard in range(20):
        if moves <= 0 or side.hp <= 0 or foe.hp <= 0:
            break

        # spend a move to evolve, or to boost an already-evolved monster
        if pol.use_berries and side.berries >= rules.BERRIES_TO_EVOLVE:
            # evolve down the list, then spend later berries boosting the first
            order = (pol.evolve_order or
                     ((pol.evolve_target,) if pol.evolve_target
                      else tuple(side.mons)))
            tgt = next((t for t in order
                        if t in side.mons and not side.evolved[t]), None)
            if tgt is None:
                tgt = next((t for t in order if t in side.mons), None)
            if tgt in side.mons and (not side.evolved[tgt] or rules.ALLOW_BOOST):
                side.berries -= rules.BERRIES_TO_EVOLVE
                if not side.evolved[tgt]:
                    side.evolved[tgt] = True
                    st[side.label + '/evolve_' + side.mons[tgt].name] += 1
                else:
                    side.mana[tgt] += rules.BOOST_MANA
                    st[side.label + '/boosts'] += 1
                _log('evolve', side=side, foe=foe, g=g, target=tgt,
                     boost=side.evolved[tgt] and st[side.label + '/boosts'])
                if rules.EVOLVE_COSTS_MOVE:
                    moves -= 1
                ctx = valuation(side, foe, pol)
                fire_all(side, foe, g, st)
                continue

        # only pay for setup ranking when a setup could actually be played
        want = (moves >= 2 and pol.setup and rules.ALLOW_NON_MATCHING_SWAP)
        matches, setups = g.all_swaps(ctx.rank, want_setups=want)
        if not matches and not setups:
            matches, setups = g.all_swaps(ctx.rank, want_setups=True)

        st[side.label + '/move_slots'] += 1
        st[side.label + '/match_options'] += len(matches)
        if not matches:
            st[side.label + '/no_match_slots'] += 1
        for c in side.mons:
            if any(c in cl for _, cl, _ in matches):
                st[side.label + '/avail_' + c] += 1

        picker = (choose_move_self_first if pol.mode == 'self_first'
                  else choose_move)
        args = (side, foe, g, ctx, pol, matches, setups,
                extras < rules.EXTRA_MOVES_PER_TURN, moves)
        kind, mv, was_ruin = (picker(*args, ruin_used)
                              if pol.mode == 'self_first' else picker(*args))
        if was_ruin:
            ruin_used += 1
            st[side.label + '/ruin_matches'] += 1
        if kind is None:
            # nothing matchable and no setup helps: shuffle a tile and move on
            mv = g.any_swap()
            if mv is None:
                break
            g.swap(mv)
            _log('idle', side=side, foe=foe, g=g, mv=mv)
            st[side.label + '/idle_moves'] += 1
            moves -= 1
            continue
        moves -= 1
        if kind == 'setup':
            g.swap(mv)
            _log('setup', side=side, foe=foe, g=g, mv=mv)
            st[side.label + '/setup_moves'] += 1
            st[side.label + ('/forced_setups' if not matches
                             else '/chosen_setups')] += 1
            continue

        cleared, longest = g.apply(mv)
        earned = longest >= 4 and extras < rules.EXTRA_MOVES_PER_TURN
        _log('match', side=side, foe=foe, g=g, mv=mv, cleared=cleared,
             longest=longest, earned=earned)
        if earned:
            extras += 1
            moves += 1
            st[side.label + '/extra_moves'] += 1
        collect(side, cleared, st)
        side.ability_longest = 0
        fire_all(side, foe, g, st)
        # a 4+ match made by an ability earns the extra move too
        if (rules.EXTRA_MOVE_FROM_ABILITY and side.ability_longest >= 4
                and extras < rules.EXTRA_MOVES_PER_TURN):
            extras += 1
            moves += 1
            st[side.label + '/extra_moves'] += 1
            st[side.label + '/extra_from_ability'] += 1

    # charges tick at the end of your own turn
    for c, m in side.mons.items():
        if m.charges_per_turn:
            side.charges[c] = min(side.charges[c] + m.charges_per_turn,
                                  m.cap(side.evolved[c]))
    _log('turn_end', side=side, foe=foe, g=g)
    for c in side.mons:
        st[side.label + '/idle_mana_' + c] += side.mana[c]


def duel(rng, my_pol, foe_pol, my_first, st=None):
    if st is None:
        st = Counter()
    g = Grid(rng)
    me = Side(MY_TEAM, rules.BASE_HP + (0 if my_first else rules.SECOND_PLAYER_HP_BONUS), 'me')
    foe = Side(FOE_TEAM, rules.BASE_HP + (rules.SECOND_PLAYER_HP_BONUS if my_first else 0), 'foe')
    order = ([(me, foe, my_pol), (foe, me, foe_pol)] if my_first
             else [(foe, me, foe_pol), (me, foe, my_pol)])
    for _turn in range(rules.MAX_TURNS):
        for (a, b, p) in order:
            play_turn(a, b, g, p, st)
            st['turns'] += 1
            if b.hp <= 0:
                st['games'] += 1
                if b is foe:
                    st['wins'] += 1
                    return 1.0
                return 0.0
    st['games'] += 1
    st['draws'] += 1
    return 0.5


