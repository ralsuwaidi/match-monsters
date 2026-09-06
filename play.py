"""Play a real game against the simulator.

    uv run python play.py new [seed]      start a game (you move first, 80 HP)
    uv run python play.py show            print the board and both sides
    uv run python play.py swap r1 c1 r2 c2   your move (adjacent tiles)
    uv run python play.py evolve red|yellow  spend a move on 4 berries
    uv run python play.py ai              let the opponent take its turn

You play Bonzumi + Sipzap. The opponent plays Pelijet + Barbenin on the
`pel_deny` policy -- the solver's strongest enemy reply.
"""
import json
import os
import random
import sys
from collections import Counter

import ai
import engine
import grid
import monsters
import rules

STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'game_state.json')
GLYPH = {'red': '🔴', 'yellow': '🟡', 'blue': '🔵',
         'green': '🟢', 'purple': '🟣', grid.BERRY: '🍒'}
FOE_POLICY = 'pel_deny'


# ------------------------------------------------------------- state io ----
def save(g, me, foe, turn_no, mover, moves, extras):
    st = g.rng.getstate()
    json.dump({
        'grid': g.g, 'rng': [st[0], list(st[1]), st[2]],
        'turn_no': turn_no, 'mover': mover, 'moves': moves, 'extras': extras,
        'me': _side(me), 'foe': _side(foe),
    }, open(STATE, 'w'))


def _side(s):
    return {'hp': s.hp, 'max_hp': s.max_hp, 'mana': s.mana,
            'evolved': s.evolved, 'charges': s.charges, 'berries': s.berries}


def _restore(team, d, label):
    s = engine.Side(team, d['hp'], label)
    s.max_hp = d['max_hp']
    s.mana, s.evolved, s.charges = d['mana'], d['evolved'], d['charges']
    s.berries = d['berries']
    return s


def load():
    d = json.load(open(STATE))
    rng = random.Random()
    rng.setstate((d['rng'][0], tuple(d['rng'][1]), d['rng'][2]))
    g = object.__new__(grid.Grid)
    g.g, g.rng, g.last_longest = d['grid'], rng, 0
    me = _restore(engine.MY_TEAM, d['me'], 'me')
    foe = _restore(engine.FOE_TEAM, d['foe'], 'foe')
    return g, me, foe, d['turn_no'], d['mover'], d['moves'], d['extras']


# --------------------------------------------------------------- render ----
def board(g):
    out = ['      ' + '  '.join(f'{c} ' for c in range(grid.W))]
    for r in range(grid.H):
        out.append(f'  {r}  ' + ' '.join(GLYPH[g.at(r, c)] for c in range(grid.W)))
    return '\n'.join(out)


def bars(s, name):
    parts = []
    for c, m in s.mons.items():
        nm = ('Bonzire' if m.name == 'Bonzumi' else
              'Ranzap' if m.name == 'Sipzap' else m.name + '+') \
            if s.evolved[c] else m.name
        b = f'{nm} {s.mana[c]}/{m.cost}'
        if m.charged:
            b += f' (honey {s.charges[c]}/{m.cap(s.evolved[c])})'
        parts.append(b)
    return (f'{name}  ❤️ {s.hp}   🍒 {s.berries}/4   ' + '   '.join(parts))


def show(g, me, foe, turn_no, mover, moves, extras):
    print()
    print(bars(foe, 'OPPONENT (Pelijet + Barbenin)'))
    print()
    print(board(g))
    print()
    print(bars(me, 'YOU      (Bonzumi + Sipzap)  '))
    print()
    who = 'YOUR' if mover == 'me' else "OPPONENT'S"
    print(f'turn {turn_no} — {who} move.  moves left this turn: {moves}'
          + ('   (extra move already used)' if extras else ''))
    if mover == 'me':
        m, s = g.all_swaps({t: 1.0 for t in grid.ALL}, want_setups=True)
        if not m:
            print('   no match is available — you can only reposition a tile')
        else:
            cols = sorted({t for _mv, cl, _l in m for t in cl})
            print('   matchable right now: ' + ' '.join(GLYPH[t] for t in cols))


def tiles(d):
    return ' '.join(f'{n}{GLYPH[t]}' for t, n in sorted(d.items(), key=lambda x: -x[1]))


# ------------------------------------------------------------- commands ----
def cmd_new(seed):
    rng = random.Random(seed)
    g = grid.Grid(rng)
    me = engine.Side(engine.MY_TEAM, rules.BASE_HP, 'me')
    foe = engine.Side(engine.FOE_TEAM,
                      rules.BASE_HP + rules.SECOND_PLAYER_HP_BONUS, 'foe')
    save(g, me, foe, 1, 'me', rules.MOVES_PER_TURN, 0)
    show(g, me, foe, 1, 'me', rules.MOVES_PER_TURN, 0)


def _end_turn(g, me, foe, turn_no, mover):
    side = me if mover == 'me' else foe
    for c, m in side.mons.items():
        if m.charges_per_turn:
            side.charges[c] = min(side.charges[c] + m.charges_per_turn,
                                  m.cap(side.evolved[c]))
    nxt = 'foe' if mover == 'me' else 'me'
    save(g, me, foe, turn_no + (1 if nxt == 'me' else 0), nxt,
         rules.MOVES_PER_TURN, 0)


def cmd_swap(a, b, c, d):
    g, me, foe, turn_no, mover, moves, extras = load()
    if mover != 'me':
        print('not your move — run `ai` first'); return
    if abs(a - c) + abs(b - d) != 1:
        print('those tiles are not adjacent'); return
    if g.at(a, b) == g.at(c, d):
        print('those tiles are the same colour'); return
    st = Counter()
    before = {k: g.at(*k) for k in ((a, b), (c, d))}
    g.g[a][b], g.g[c][d] = g.g[c][d], g.g[a][b]
    hits, longest = g._runs_near(((a, b), (c, d)))
    if not hits:
        print(f'\n  repositioned {GLYPH[before[(a,b)]]}↔{GLYPH[before[(c,d)]]} '
              f'— no match, move spent')
        moves -= 1
    else:
        cleared, longest = g.resolve()
        print(f'\n  ✅ matched {tiles(cleared)}   (longest run {longest})')
        engine.collect(me, cleared, st)
        moves -= 1
        if longest >= 4 and extras < rules.EXTRA_MOVES_PER_TURN:
            extras += 1; moves += 1
            print('  ⭐ 4+ match — EXTRA MOVE')
        me.ability_longest = 0
        engine.fire_all(me, foe, g, st)
        for k, v in st.items():
            if '/fires_' in k:
                print(f'  💥 {k.split("_")[-1]} STRIKES ×{v}')
        if (rules.EXTRA_MOVE_FROM_ABILITY and me.ability_longest >= 4
                and extras < rules.EXTRA_MOVES_PER_TURN):
            extras += 1; moves += 1
            print('  ⭐ an ability made a 4+ match — EXTRA MOVE')
    if foe.hp <= 0:
        print('\n  🏆 YOU WIN'); save(g, me, foe, turn_no, 'me', 0, extras); return
    if moves <= 0:
        _end_turn(g, me, foe, turn_no, 'me')
        g, me, foe, turn_no, mover, moves, extras = load()
    else:
        save(g, me, foe, turn_no, 'me', moves, extras)
    show(g, me, foe, turn_no, mover if moves else 'foe', moves, extras)


def cmd_evolve(colour):
    g, me, foe, turn_no, mover, moves, extras = load()
    if mover != 'me':
        print('not your move'); return
    if me.berries < rules.BERRIES_TO_EVOLVE:
        print(f'only {me.berries}/4 berries'); return
    me.berries -= rules.BERRIES_TO_EVOLVE
    if not me.evolved[colour]:
        me.evolved[colour] = True
        print(f'\n  ✨ EVOLVED {me.mons[colour].name}')
    else:
        me.mana[colour] += rules.BOOST_MANA
        print(f'\n  ⚡ BOOSTED {me.mons[colour].name} +{rules.BOOST_MANA} mana')
    moves -= 1
    st = Counter()
    engine.fire_all(me, foe, g, st)
    if moves <= 0:
        _end_turn(g, me, foe, turn_no, 'me')
        g, me, foe, turn_no, mover, moves, extras = load()
    else:
        save(g, me, foe, turn_no, 'me', moves, extras)
    show(g, me, foe, turn_no, mover if moves else 'foe', moves, extras)


def cmd_ai():
    g, me, foe, turn_no, mover, moves, extras = load()
    if mover != 'foe':
        print('it is your move'); return
    st = Counter()
    log = []

    def trace(kind, **kw):
        if kw.get('side') is not foe:
            return
        if kind == 'match':
            log.append(f'  matched {tiles(kw["cleared"])} at '
                       f'{kw["mv"][:2]}↔{kw["mv"][2:]}'
                       + ('   ⭐ EXTRA MOVE' if kw['earned'] else ''))
        elif kind == 'setup':
            log.append(f'  repositioned {kw["mv"][:2]}↔{kw["mv"][2:]} (no match)')
        elif kind == 'evolve':
            log.append(f'  ✨ {"BOOSTED" if kw["boost"] else "EVOLVED"} '
                       f'{foe.mons[kw["target"]].name}')
        elif kind == 'fire':
            ab = kw['ab']
            bits = [f'{ab.damage} damage'] if ab.damage else []
            if ab.convert_n:
                bits.append(f'converts {ab.convert_n} tiles to Water')
            if ab.drain:
                bits.append(f'drains {ab.drain} mana from EACH of your monsters')
            log.append(f'  💥 {kw["mon"].name} STRIKES: ' + '; '.join(bits))
    engine.TRACE = trace
    engine.play_turn(foe, me, g, ai.FOE_POLICIES[FOE_POLICY], st)
    engine.TRACE = None
    print('\nOPPONENT:')
    for line in log:
        print(line)
    if me.hp <= 0:
        print('\n  💀 YOU LOSE'); save(g, me, foe, turn_no, 'foe', 0, 0); return
    _end_turn(g, me, foe, turn_no, 'foe')
    g, me, foe, turn_no, mover, moves, extras = load()
    show(g, me, foe, turn_no, mover, moves, extras)


# ------------------------------------------------------- interactive mode ----
LOGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'games')
_log = []
_logfile = None


def open_log(seed, policy):
    """Open the transcript up front and flush every line, so the log is
    complete even if the game is quit or crashes part way through."""
    global _logfile
    os.makedirs(LOGDIR, exist_ok=True)
    import time
    path = os.path.join(LOGDIR, 'game-%s-seed%d.log'
                        % (time.strftime('%Y%m%d-%H%M%S'), seed))
    _logfile = open(path, 'w', buffering=1)   # line buffered
    return path


def out(msg=''):
    print(msg)
    _log.append(msg)
    if _logfile:
        _logfile.write(msg + '\n')
        _logfile.flush()


def legal(g, me):
    m, _s = g.all_swaps({t: 1.0 for t in grid.ALL}, want_setups=True)
    rows = []
    for mv, cl, ln in m:
        note = ''
        for c, mon in me.mons.items():
            if c in cl:
                tot = me.mana[c] + cl[c]
                if tot >= mon.cost:
                    if mon.charged:
                        ch = me.charges[c]
                        dmg = mon.ability(me.evolved[c], ch).damage
                        note = f'  -> FIRES {mon.name} at honey {ch} = {dmg} dmg'
                    else:
                        note = f'  -> FIRES {mon.name}'
                else:
                    note = f'  -> {mon.name} {tot}/{mon.cost}'
        rows.append((mv, cl, ln, note))
    return rows


def show_moves(g, me):
    rows = legal(g, me)
    if not rows:
        out('  no match available -- you can only reposition a tile')
        return
    out('  legal matches:')
    for mv, cl, ln, note in rows:
        out('    %d %d %d %d   %s%s%s' % (
            mv[0], mv[1], mv[2], mv[3],
            ' '.join('%d%s' % (n, GLYPH[t]) for t, n in
                     sorted(cl.items(), key=lambda x: -x[1])),
            '   (run %d -> EXTRA MOVE)' % ln if ln >= 4 else '', note))


HELP = """  commands
    r1 c1 r2 c2   swap two adjacent tiles, e.g.  3 4 3 5
    m             list every legal match
    e red         spend a move + 4 berries to evolve/boost Bonzumi
    e yellow      same for Sipzap
    b             reprint the board
    q             quit and save the log
"""


def interactive(seed, foe_policy):
    from collections import Counter
    path = open_log(seed, foe_policy)
    print('logging to %s  (updated after every move)\n' % path)
    rng = random.Random(seed)
    g = grid.Grid(rng)
    me = engine.Side(engine.MY_TEAM, rules.BASE_HP, 'me')
    foe = engine.Side(engine.FOE_TEAM,
                      rules.BASE_HP + rules.SECOND_PLAYER_HP_BONUS, 'foe')

    out('MATCH MONSTERS -- you vs the simulator')
    out('  seed %d   enemy policy %s' % (seed, foe_policy))
    out('  rules: board %dx%d, berries %.0f%%, mana carryover %s, HP %d (+%d 2nd), '
        'boost %d, berry cap %d'
        % (grid.W, grid.H, 100 * grid.BERRY_WEIGHT,
           'on' if rules.MANA_CARRYOVER else 'off', rules.BASE_HP,
           rules.SECOND_PLAYER_HP_BONUS, rules.BOOST_MANA, rules.BERRY_CAP))
    out('  you are Bonzumi (Fire 8) + Sipzap (Electric 4), 80 HP, you move first')
    out()
    out(HELP)

    turn = 1
    while True:
        moves, extras = rules.MOVES_PER_TURN + me.extra_moves_pending, 0
        me.extra_moves_pending = 0
        out('=' * 60)
        out('TURN %d -- your move' % turn)
        for line in render(g, me, foe).split('\n'):
            out(line)
        while moves > 0:
            show_moves(g, me)
            out('  moves left: %d%s' % (moves, '  (extra used)' if extras else ''))
            try:
                raw = input('> ').strip()
            except EOFError:
                raw = 'q'
            _log.append('> ' + raw)
            if not raw:
                continue
            if raw[0] == 'q':
                return finish(seed, None)
            if raw[0] == 'b':
                for line in render(g, me, foe).split('\n'):
                    out(line)
                continue
            if raw[0] == 'm':
                continue
            if raw[0] == '?':
                out(HELP)
                continue
            if raw[0] == 'e':
                colour = 'yellow' if 'y' in raw else 'red'
                if me.berries < rules.BERRIES_TO_EVOLVE:
                    out('  only %d/4 berries' % me.berries)
                    continue
                me.berries -= rules.BERRIES_TO_EVOLVE
                if not me.evolved[colour]:
                    me.evolved[colour] = True
                    out('  EVOLVED %s' % me.mons[colour].name)
                else:
                    me.mana[colour] += rules.BOOST_MANA
                    out('  BOOSTED %s +%d mana' % (me.mons[colour].name,
                                                   rules.BOOST_MANA))
                moves -= 1
                st = Counter()
                engine.fire_all(me, foe, g, st)
                report_fires(st, 'me')
                if foe.hp <= 0:
                    return finish(seed, 'you win')
                continue
            nums = [int(x) for x in raw.replace(',', ' ').split() if
                    x.lstrip('-').isdigit()]
            if len(nums) != 4:
                out('  give four numbers: r1 c1 r2 c2')
                continue
            a, b, c, d = nums
            if not (0 <= a < grid.H and 0 <= c < grid.H
                    and 0 <= b < grid.W and 0 <= d < grid.W):
                out('  off the board')
                continue
            if abs(a - c) + abs(b - d) != 1:
                out('  those tiles are not adjacent')
                continue
            if g.at(a, b) == g.at(c, d):
                out('  those tiles are the same colour')
                continue
            st = Counter()
            g.g[a][b], g.g[c][d] = g.g[c][d], g.g[a][b]
            hits, _ln = g._runs_near(((a, b), (c, d)))
            if not hits:
                out('  repositioned (%d,%d)<->(%d,%d) -- no match, move spent'
                    % (a, b, c, d))
                moves -= 1
            else:
                cleared, ln = g.resolve()
                out('  matched %s  (run %d)' % (tiles(cleared), ln))
                engine.collect(me, cleared, st)
                moves -= 1
                if ln >= 4 and extras < rules.EXTRA_MOVES_PER_TURN:
                    extras += 1
                    moves += 1
                    out('  4+ match -- EXTRA MOVE')
                me.ability_longest = 0
                engine.fire_all(me, foe, g, st)
                report_fires(st, 'me')
                if (rules.EXTRA_MOVE_FROM_ABILITY and me.ability_longest >= 4
                        and extras < rules.EXTRA_MOVES_PER_TURN):
                    extras += 1
                    moves += 1
                    out('  an ability made a 4+ match -- EXTRA MOVE')
            if foe.hp <= 0:
                return finish(seed, 'you win')
            for line in render(g, me, foe).split('\n'):
                out(line)

        for cc, mon in me.mons.items():
            if mon.charges_per_turn:
                me.charges[cc] = min(me.charges[cc] + mon.charges_per_turn,
                                     mon.cap(me.evolved[cc]))

        out('-' * 60)
        out('TURN %d -- opponent (%s)' % (turn, foe_policy))
        st = Counter()
        lines = []

        def trace(kind, **kw):
            if kw.get('side') is not foe:
                return
            if kind == 'match':
                lines.append('  matched %s at (%d,%d)<->(%d,%d)%s' % (
                    tiles(kw['cleared']), *kw['mv'],
                    '   EXTRA MOVE' if kw['earned'] else ''))
            elif kind == 'setup':
                lines.append('  repositioned (%d,%d)<->(%d,%d)' % kw['mv'])
            elif kind == 'evolve':
                lines.append('  %s %s' % ('BOOSTED' if kw['boost'] else 'EVOLVED',
                                          foe.mons[kw['target']].name))
            elif kind == 'fire':
                ab = kw['ab']
                bits = ['%d damage' % ab.damage] if ab.damage else []
                if ab.convert_n:
                    bits.append('converts %d tiles to Water' % ab.convert_n)
                if ab.drain:
                    bits.append('drains %d mana from EACH of your monsters'
                                % ab.drain)
                lines.append('  %s STRIKES: %s' % (kw['mon'].name, '; '.join(bits)))
        engine.TRACE = trace
        try:
            engine.play_turn(foe, me, g, ai.FOE_POLICIES[foe_policy], st)
        finally:
            engine.TRACE = None
        for line in lines:
            out(line)
        if me.hp <= 0:
            for line in render(g, me, foe).split('\n'):
                out(line)
            return finish(seed, 'you lose')
        for cc, mon in foe.mons.items():
            if mon.charges_per_turn:
                foe.charges[cc] = min(foe.charges[cc] + mon.charges_per_turn,
                                      mon.cap(foe.evolved[cc]))
        turn += 1


def render(g, me, foe):
    return '\n'.join(['', bars(foe, 'OPPONENT'), '', board(g), '',
                       bars(me, 'YOU     '), ''])


def report_fires(st, lbl):
    for k, v in st.items():
        if k.startswith(lbl + '/fires_'):
            out('  %s STRIKES x%d' % (k.split('_')[-1], v))


def finish(seed, result):
    if result:
        out('')
        out('RESULT: %s' % result.upper())
    path = _logfile.name if _logfile else '(none)'
    if _logfile:
        _logfile.close()
    print()
    print('log written to %s' % path)
    print('paste that file back to Claude to review the game.')


if __name__ == '__main__':

    cmd = sys.argv[1] if len(sys.argv) > 1 else 'play'
    if cmd == 'play':
        sd = int(sys.argv[2]) if len(sys.argv) > 2 else random.randrange(10 ** 6)
        pol = sys.argv[3] if len(sys.argv) > 3 else FOE_POLICY
        interactive(sd, pol)
    elif cmd == 'new':
        cmd_new(int(sys.argv[2]) if len(sys.argv) > 2 else random.randrange(10**6))
    elif cmd == 'swap':
        cmd_swap(*[int(x) for x in sys.argv[2:6]])
    elif cmd == 'evolve':
        cmd_evolve(sys.argv[2])
    elif cmd == 'ai':
        cmd_ai()
    else:
        show(*load())
