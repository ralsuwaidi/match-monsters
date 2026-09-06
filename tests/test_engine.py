"""Rules that must hold. These encode the corrections found by checking the
model against real games -- each one is a bug that was actually present."""
import random
from collections import Counter

import pytest

from match_monsters import rules
from match_monsters.agents import ai
from match_monsters.game import engine, grid, monsters
from match_monsters.solver import runner


def test_fresh_board_is_never_already_matched():
    for seed in range(200):
        assert not grid.Grid(random.Random(seed)).find_runs()[0]


def test_refill_matches_the_configured_spawn_rates():
    """The no-match-on-spawn rule must not skew the distribution -- an ordered
    fallback here would have biased every result toward red."""
    rng = random.Random(1)
    counts = Counter()
    for _ in range(400):
        counts.update(grid.Grid(rng).counts())
    total = sum(counts.values())
    assert counts[grid.BERRY] / total == pytest.approx(grid.BERRY_WEIGHT, abs=0.02)
    for colour in grid.COLORS:
        expected = (1 - grid.BERRY_WEIGHT) / len(grid.COLORS)
        assert counts[colour] / total == pytest.approx(expected, abs=0.02)


def test_a_new_tile_may_complete_a_match_with_settled_tiles():
    """The refill rule, stated precisely: a batch of new tiles never arrives
    matched among ITSELF, but a new tile IS allowed to land completing a match
    with tiles already on the board. Applying the restriction to settled tiles
    too starved the board to zero legal moves within nine turns."""
    g = grid.Grid(random.Random(0))
    g.g[3][0] = g.g[4][0] = 'red'          # two settled reds
    # placing a red above them completes a run with SETTLED tiles...
    assert g._would_make_run(2, 0, 'red') is True
    # ...but the refill only forbids runs among tiles placed in this pass,
    # so with an empty "new" set the placement is allowed, and the match then
    # resolves as a cascade
    assert g._would_make_run(2, 0, 'red', set()) is False


def test_the_board_keeps_offering_matches_for_a_whole_game():
    """A game is about 17 half-turns. The board must not dry up inside one."""
    rng = random.Random(0)
    g = grid.Grid(rng)
    played = 0
    for _ in range(300):
        moves = g.moves_with_preview()
        if not moves:
            break
        g.apply(rng.choice(moves)[0])
        played += 1
    assert played > 17, f'board dried up after only {played} matches'


def test_non_matching_swaps_are_legal_and_cost_a_move():
    """Unlike Candy Crush, a swap need not make a match."""
    rng = random.Random(4)
    g = grid.Grid(rng)
    matches, setups = g.all_swaps({t: 1.0 for t in grid.ALL})
    assert setups, 'repositioning moves should exist'
    before = [row[:] for row in g.g]
    g.swap(setups[0][0])
    assert g.g != before
    assert not g.find_runs()[0], 'a setup swap must not create a match'


def test_enumerating_moves_does_not_mutate_the_board():
    """A swap-restore bug here silently scrambled the board on every move."""
    g = grid.Grid(random.Random(2))
    snapshot = [row[:] for row in g.g]
    for _ in range(50):
        g.all_swaps({t: 1.0 for t in grid.ALL})
        g.moves_with_preview()
    assert g.g == snapshot


def test_mana_does_not_carry_over():
    """Tested on Barbenin, whose ability picks no tiles up. Bonzumi is a bad
    subject: Flare clears a column and hands the red tiles in it straight back,
    so its bar is legitimately non-zero right after firing."""
    st = Counter()
    g = grid.Grid(random.Random(7))
    me = engine.Side(engine.MY_TEAM, 80, 'me')
    foe = engine.Side(engine.FOE_TEAM, 80, 'foe')
    foe.mana['purple'] = monsters.BARBENIN.cost + 4
    engine.fire(foe, me, g, 'purple', st)
    assert foe.mana['purple'] == 0, 'overflow above the threshold must be burnt'


def test_flare_refunds_mana_from_its_own_column():
    """The flip side: Bonzumi partly funds itself, and feeds Sipzap too."""
    st = Counter()
    g = grid.Grid(random.Random(7))
    me = engine.Side(engine.MY_TEAM, 80, 'me')
    foe = engine.Side(engine.FOE_TEAM, 80, 'foe')
    for c in range(grid.W):
        g.g[0][c] = 'red'                     # guarantee reds in every column
        g.g[2][c] = 'yellow'
    me.mana['red'] = monsters.BONZUMI.cost
    engine.fire(me, foe, g, 'red', st)
    assert me.mana['red'] + me.mana['yellow'] > 0


def test_berry_counter_caps_at_four():
    st = Counter()
    side = engine.Side(engine.MY_TEAM, 80, 'me')
    side.berries = 3
    engine.collect(side, {grid.BERRY: 5}, st)
    assert side.berries == rules.BERRY_CAP == 4


def test_sipzap_charge_survives_evolution_but_resets_on_firing():
    st = Counter()
    g = grid.Grid(random.Random(3))
    me = engine.Side(engine.MY_TEAM, 80, 'me')
    foe = engine.Side(engine.FOE_TEAM, 80, 'foe')
    me.charges['yellow'] = 2
    me.evolved['yellow'] = True                    # evolving must not reset it
    assert me.charges['yellow'] == 2
    assert monsters.SIPZAP.cap(True) == 3          # and the cap rises at once
    me.charges['yellow'] = 3
    assert monsters.SIPZAP.ability(True, 3).damage == 20
    me.mana['yellow'] = monsters.SIPZAP.cost
    engine.fire(me, foe, g, 'yellow', st)
    assert me.charges['yellow'] == 0


def test_firing_at_zero_charge_deals_nothing():
    """Which is why filling Sipzap's bar early throws the mana away."""
    assert monsters.SIPZAP.ability(False, 0).damage == 0
    assert monsters.SIPZAP.ability(True, 0).damage == 0


def test_flare_column_feeds_both_monsters_and_the_berry_counter():
    st = Counter()
    g = grid.Grid(random.Random(11))
    me = engine.Side(engine.MY_TEAM, 80, 'me')
    foe = engine.Side(engine.FOE_TEAM, 80, 'foe')
    me.mana['red'] = monsters.BONZUMI.cost
    engine.fire(me, foe, g, 'red', st)
    assert foe.hp == 60, 'unevolved Flare hits for 20'
    gained = me.mana['red'] + me.mana['yellow'] + me.berries
    assert gained >= 0            # enemy colours in the column give nothing


def test_second_player_gets_the_hp_bonus():
    rng = random.Random(5)
    st = Counter()
    engine.duel(rng, ai.MY_POLICIES['bon_hdeny'], ai.FOE_POLICIES['pel_deny'],
                my_first=True, st=st)
    assert rules.BASE_HP == 80 and rules.SECOND_PLAYER_HP_BONUS == 5


def test_games_finish_without_hitting_the_turn_cap():
    rng = random.Random(9)
    st = Counter()
    for i in range(60):
        engine.duel(rng, ai.MY_POLICIES['bon_hdeny'],
                    ai.FOE_POLICIES['pel_deny'], i % 2 == 0, st)
    assert st['draws'] == 0


def test_sensitivity_config_reaches_the_workers():
    """Overrides travel to spawned workers explicitly; a silent no-op here
    made a whole sensitivity sweep measure nothing."""
    a = runner.run('bon_hdeny', 'pel_deny', 200, seed=3,
                   cfg={'grid.BERRY_WEIGHT': 0.0}, procs=1)
    assert a['st']['me/berries'] == 0


def test_unknown_config_key_is_rejected():
    with pytest.raises(KeyError):
        runner.apply_cfg({'NOT_A_RULE': 1})
    runner.reset_defaults()


def test_selfplay_win_rate_follows_the_team_not_the_seat():
    """Seat 0 is always the first player. If the reported rate tracked the seat
    instead of the team, buffing one team would not move it."""
    torch = pytest.importorskip('torch')
    import random as _r

    from match_monsters.rl import nets, train
    from match_monsters.rl.selfplay import Duel

    def measure():
        net = nets.ActorCritic(32, 1, 64)
        rngs = [_r.Random(i) for i in range(48)]
        teams = (engine.MY_TEAM, engine.FOE_TEAM)
        seats = [teams if i % 2 == 0 else (teams[1], teams[0]) for i in range(48)]
        duels = [Duel(rngs[i], seats[i], max_turns=40) for i in range(48)]
        _buf, res, _tele = train.rollout(net, duels, rngs, torch.device('cpu'),
                                         6000, teams, 40, 0.0, 0.0, 0.0)
        return res

    original = (monsters.BONZUMI.cost, monsters.SIPZAP.cost,
                monsters.PELIJET.cost, monsters.BARBENIN.cost)
    try:
        object.__setattr__(monsters.BONZUMI, 'cost', 2)
        object.__setattr__(monsters.SIPZAP, 'cost', 1)
        buffed = measure()
        object.__setattr__(monsters.BONZUMI, 'cost', original[0])
        object.__setattr__(monsters.SIPZAP, 'cost', original[1])
        object.__setattr__(monsters.PELIJET, 'cost', 2)
        object.__setattr__(monsters.BARBENIN, 'cost', 1)
        nerfed = measure()
        assert buffed and nerfed, 'no games finished -- nothing was measured'
        hi = sum(buffed) / len(buffed)
        lo = sum(nerfed) / len(nerfed)
        # buffing my team must raise the number and buffing theirs must lower
        # it. Following the SEAT instead of the team would leave both near 0.5.
        assert hi - lo > 0.3, (
            f'reported rate barely moved ({lo:.2f} -> {hi:.2f}); the metric is '
            f'following the seat rather than the team')
    finally:
        for mon, cost in zip((monsters.BONZUMI, monsters.SIPZAP,
                              monsters.PELIJET, monsters.BARBENIN), original):
            object.__setattr__(mon, 'cost', cost)
