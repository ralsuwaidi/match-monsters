"""Detailed head-to-head between two policies, for the UI and the CLI."""

import monsters
import runner

MY = (monsters.BONZUMI, monsters.SIPZAP)
FOE = (monsters.PELIJET, monsters.BARBENIN)


def head_to_head(my_pol, foe_pol, n=2000, seed=777, cfg=None):
    res = runner.run(my_pol, foe_pol, n, seed=seed, cfg=cfg)
    st = res['st']
    g = max(1, st['games'])
    lo, hi = runner.wilson(res['wr'], res['n'])
    out = {
        'win': 100 * res['wr'], 'lo': 100 * lo, 'hi': 100 * hi, 'n': res['n'],
        'first': 100 * res['wr_first'], 'second': 100 * res['wr_second'],
        'rounds': st['turns'] / g / 2, 'monsters': [],
    }
    for lbl, team, other in (('me', MY, 'foe'), ('foe', FOE, 'me')):
        earned = sum(st[f'{lbl}/mana_{m.color}'] for m in team) / g
        spent = sum(st[f'{lbl}/spent_{m.name}'] for m in team) / g
        waste = sum(st[f'{lbl}/wasted_{m.name}'] for m in team) / g
        drained = st[f'{other}/drained'] / g
        out[lbl] = {
            'damage': sum(st[f'{lbl}/dmg_{m.name}'] for m in team) / g,
            'earned': earned, 'spent': spent, 'wasted': waste,
            'drained': drained,
            'usable_pct': 100 * spent / earned if earned else 0,
            'dmg_per_mana': (sum(st[f'{lbl}/dmg_{m.name}'] for m in team) / g
                             / spent if spent else 0),
            'berries': st[f'{lbl}/berries'] / g,
            'setups': st[f'{lbl}/setup_moves'] / g,
            'ruin': st[f'{lbl}/ruin_matches'] / g,
            'extra': st[f'{lbl}/extra_moves'] / g,
        }
        for m in team:
            out['monsters'].append({
                'side': 'mine' if lbl == 'me' else 'theirs',
                'monster': m.name, 'cost': m.cost,
                'strikes': st[f'{lbl}/fires_{m.name}'] / g,
                'damage': st[f'{lbl}/dmg_{m.name}'] / g,
                'mana spent': st[f'{lbl}/spent_{m.name}'] / g,
                'mana burnt': st[f'{lbl}/wasted_{m.name}'] / g,
                'evolutions': st[f'{lbl}/evolve_{m.name}'] / g,
            })
    return out
