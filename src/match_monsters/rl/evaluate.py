"""Evaluate the co-evolved networks: against each other, and against the
hand-tuned heuristics, so you can see which side actually got better.

    uv run --extra rl python eval_nn.py --games 400
"""
import argparse
import os
import random
from collections import Counter

import numpy as np
import torch

from match_monsters.agents import ai
from match_monsters.game import engine
from match_monsters.rl import nets
from match_monsters import rules
from match_monsters.rl import selfplay
from match_monsters.rl.selfplay import Duel, N_SWAP, SWAP_IX


class NetAgent:
    def __init__(self, net, device, greedy=True):
        self.net, self.device, self.greedy = net, device, greedy
        self.name = 'net'

    @torch.no_grad()
    def act_batch(self, duels):
        obs = [d.observe() for d in duels]
        masks = np.asarray([d.legal_mask() for d in duels])
        b = torch.as_tensor(np.asarray([o[0] for o in obs]), device=self.device)
        s = torch.as_tensor(np.asarray([o[1] for o in obs]), device=self.device)
        m = torch.as_tensor(masks, device=self.device)
        logits, _ = self.net(b, s, m)
        if self.greedy:
            return logits.argmax(-1).cpu().numpy()
        return torch.distributions.Categorical(logits=logits).sample().cpu().numpy()


class SearchAgent:
    """Use the network as an EVALUATOR rather than as a policy.

    The hand-tuned opponent is a one-ply greedy scorer: it applies each legal
    move and rates the result with a formula somebody wrote by hand. This does
    exactly the same thing with a value function that was trained on real
    outcomes instead. Same algorithm, better evaluation -- which is the shape of
    result that has beaten hand-written evaluators in every game it has been
    tried on.

    The refill after a match is random, so each candidate is rolled out
    `samples` times and averaged: one lookahead is one sample of many.
    """

    def __init__(self, net, device, samples=2, top_k=8, prior=0.35):
        self.net, self.device = net, device
        self.samples, self.top_k, self.prior = samples, top_k, prior
        self.name = 'search'

    @torch.no_grad()
    def act_batch(self, duels):
        return np.asarray([self._one(d) for d in duels])

    @torch.no_grad()
    def _one(self, d):
        board, scal = d.observe()
        mask = d.legal_mask()
        logits, _v = self.net(
            torch.as_tensor(board[None], device=self.device),
            torch.as_tensor(scal[None], device=self.device),
            torch.as_tensor(mask[None], device=self.device))
        logp = torch.log_softmax(logits[0], -1).cpu().numpy()
        legal = np.flatnonzero(mask)
        if len(legal) == 1:
            return int(legal[0])
        # the policy proposes; only its best few are actually searched
        cand = legal[np.argsort(-logp[legal])[:self.top_k]]

        states, owners = [], []
        for a in cand:
            for _ in range(self.samples):
                c = d.clone()
                side = c.active
                c.step(int(a))
                states.append(c)
                owners.append(side)
        boards = np.asarray([s.observe()[0] for s in states])
        scals = np.asarray([s.observe()[1] for s in states])
        masks = np.asarray([s.legal_mask() for s in states])
        _lg, vals = self.net(torch.as_tensor(boards, device=self.device),
                             torch.as_tensor(scals, device=self.device),
                             torch.as_tensor(masks, device=self.device))
        vals = vals.cpu().numpy()
        best, best_score = int(cand[0]), -1e9
        for i, a in enumerate(cand):
            chunk = slice(i * self.samples, (i + 1) * self.samples)
            v = 0.0
            for j, st in zip(range(*chunk.indices(len(states))), states[chunk]):
                # the value head speaks for whoever is on move in that state,
                # so flip it when control has passed to the opponent
                sign = 1.0 if st.active == owners[j] else -1.0
                if st.done:
                    r = st.result if owners[j] == 0 else 1.0 - st.result
                    v += 2 * r - 1
                else:
                    v += sign * float(vals[j])
            v /= self.samples
            score = v + self.prior * logp[a]
            if score > best_score:
                best, best_score = int(a), score
        return best


class HeuristicAgent:
    """The hand-tuned policy, driven one decision at a time so it can share the
    same game loop as the networks."""

    def __init__(self, policy):
        self.pol = policy
        self.name = policy.name

    def act_batch(self, duels):
        return np.asarray([self._one(d) for d in duels])

    def _one(self, d):
        side, foe = d.sides[d.active], d.sides[1 - d.active]
        pol = self.pol
        ctx = ai.valuation(side, foe, pol)
        if pol.use_berries and side.berries >= rules.BERRIES_TO_EVOLVE:
            order = (pol.evolve_order or
                     ((pol.evolve_target,) if pol.evolve_target
                      else tuple(side.mons)))
            tgt = next((t for t in order
                        if t in side.mons and not side.evolved[t]), None)
            if tgt is None:
                tgt = next((t for t in order if t in side.mons), None)
            if tgt is not None and (not side.evolved[tgt] or rules.ALLOW_BOOST):
                return N_SWAP + list(side.mons).index(tgt)
        want = (d.moves >= 2 and pol.setup and rules.ALLOW_NON_MATCHING_SWAP)
        matches, setups = d.g.all_swaps(ctx.rank, want_setups=want)
        if not matches and not setups:
            matches, setups = d.g.all_swaps(ctx.rank, want_setups=True)
        picker = (ai.choose_move_self_first if pol.mode == 'self_first'
                  else ai.choose_move)
        a = (side, foe, d.g, ctx, pol, matches, setups,
             d.extras < rules.EXTRA_MOVES_PER_TURN, d.moves)
        kind, mv, _ = (picker(*a, 0) if pol.mode == 'self_first' else picker(*a))
        if mv is None:
            return 0
        return SWAP_IX.get(tuple(mv), 0)


def play(agent0, agent1, n_games, seed=0, batch=128, max_turns=60, teams=None):
    teams = teams or (engine.MY_TEAM, engine.FOE_TEAM)
    rng = random.Random(seed)
    results = []
    while len(results) < n_games:
        k = min(batch, n_games - len(results))
        duels = [Duel(random.Random(rng.randrange(1 << 30)), teams,
                      max_turns=max_turns) for _ in range(k)]
        while any(not d.done for d in duels):
            for side, agent in ((0, agent0), (1, agent1)):
                live = [d for d in duels if not d.done and d.active == side]
                if not live:
                    continue
                acts = agent.act_batch(live)
                for d, a in zip(live, acts):
                    d.step(int(a))
        results.extend(d.result for d in duels)
    return float(np.mean(results)), len(results)


def wilson(p, n, z=1.96):
    import math
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(max(0.0, p * (1 - p) / n) + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def behaviour(agent0, agent1, n_games=300, seed=5, max_turns=60):
    """Play the two agents and report what each SIDE actually did.

    A win rate says who won; this says why. Seat 0 is agent0 throughout, so the
    two columns are directly comparable."""
    teams = (engine.MY_TEAM, engine.FOE_TEAM)
    rng = random.Random(seed)
    rows = {0: Counter(), 1: Counter()}
    games = 0
    results = []
    # record the charge level whenever a charge monster fires -- firing Sipzap
    # at honey 0 deals nothing, so this is where conversion efficiency lives
    honey = {0: Counter(), 1: Counter()}
    orig_fire = engine.fire

    def spy_fire(side, foe, g, colour, st):
        mon = side.mons[colour]
        if mon.charged:
            honey[0 if side.label == 'me' else 1][side.charges[colour]] += 1
        return orig_fire(side, foe, g, colour, st)
    engine.fire = spy_fire
    while games < n_games:
        k = min(128, n_games - games)
        duels = [Duel(random.Random(rng.randrange(1 << 30)), teams,
                      max_turns=max_turns) for _ in range(k)]
        swaps = {id(d): [0, 0] for d in duels}
        while any(not d.done for d in duels):
            for side, agent in ((0, agent0), (1, agent1)):
                live = [d for d in duels if not d.done and d.active == side]
                if not live:
                    continue
                before = {id(d): (d.n_swaps, d.n_matched) for d in live}
                acts = agent.act_batch(live)
                for d, a in zip(live, acts):
                    d.step(int(a))
                    s0, m0 = before[id(d)]
                    swaps[id(d)][side] += 0   # placeholder, totals read below
        for d in duels:
            games += 1
            results.append(d.result)
            for side in (0, 1):
                lbl = 'me' if side == 0 else 'foe'
                r = rows[side]
                r['damage'] += sum(v for k2, v in d.st.items()
                                   if k2.startswith(lbl + '/dmg_'))
                r['fires'] += sum(v for k2, v in d.st.items()
                                  if k2.startswith(lbl + '/fires_'))
                r['evolves'] += sum(v for k2, v in d.st.items()
                                    if k2.startswith(lbl + '/evolve_'))
                r['boosts'] += d.st[lbl + '/boosts']
                r['berries'] += d.st[lbl + '/berries']
                r['wasted_mana'] += sum(v for k2, v in d.st.items()
                                        if k2.startswith(lbl + '/wasted_'))
                r['mana'] += sum(v for k2, v in d.st.items()
                                 if k2.startswith(lbl + '/mana_'))
                r['denied'] += d.st[lbl + '/denied']
            rows[0]['turns'] += d.turn
            rows[1]['turns'] += d.turn
    engine.fire = orig_fire
    for side in (0, 1):
        for k2 in rows[side]:
            rows[side][k2] /= games
    return rows, 100 * sum(results) / games, games, honey


def team_balance(agent, n_games=2000, seed=99, max_turns=60):
    """Which TEAM is actually winning?

    Seat 0 is always the first player, so simply reporting seat 0's win rate
    conflates the team with the first-move advantage. This plays half the games
    with each team in seat 0 and reports the result by team, with the turn-order
    split broken out so the two effects can be told apart.
    """
    MINE, THEIRS = engine.MY_TEAM, engine.FOE_TEAM
    half = n_games // 2
    # mine in seat 0 -> mine moves first; result 1.0 means seat 0 won
    a, _ = play(agent, agent, half, seed=seed, max_turns=max_turns,
                teams=(MINE, THEIRS))
    # theirs in seat 0 -> mine moves second, and mine winning means seat 0 LOST
    b, _ = play(agent, agent, half, seed=seed + 1, max_turns=max_turns,
                teams=(THEIRS, MINE))
    mine_first, mine_second = a, 1.0 - b
    return {
        'mine_overall': 100 * (mine_first + mine_second) / 2,
        'mine_first': 100 * mine_first,
        'mine_second': 100 * mine_second,
        'first_move_edge': 100 * (mine_first - mine_second) / 2,
        'n': half * 2,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='checkpoints/selfplay.pt')
    ap.add_argument('--games', type=int, default=400)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--search', type=int, default=0,
                    help='also evaluate the network used as an EVALUATOR with '
                         'N refill samples per candidate move, the same way the '
                         'hand-tuned policy works but with a learned score')
    ap.add_argument('--search-top', type=int, default=8,
                    help='how many of the policy\'s best moves to search')
    ap.add_argument('--sample', action='store_true',
                    help='sample from the policy instead of taking the argmax')
    args = ap.parse_args()

    device = nets.pick_device(args.device)
    path = args.ckpt
    if os.path.isdir(path):                 # accept a directory too
        path = os.path.join(path, 'selfplay.pt')
    ck = torch.load(path, map_location=device)
    arch = ck.get('arch', {})
    A = nets.ActorCritic(**arch).to(device).eval()
    B = nets.ActorCritic(**arch).to(device).eval()
    if ck.get('shared'):
        A.load_state_dict(ck['net']); B.load_state_dict(ck['net'])
        print('shared checkpoint: the same network plays both teams')
    else:
        A.load_state_dict(ck['A']); B.load_state_dict(ck['B'])
    print(f'checkpoint {path}, iteration {ck["iter"]}, device {device}\n')

    netA = NetAgent(A, device, greedy=not args.sample)
    netB = NetAgent(B, device, greedy=not args.sample)
    hA = HeuristicAgent(ai.MY_POLICIES['bon_hdeny'])
    hB = HeuristicAgent(ai.FOE_POLICIES['pel_deny'])

    if ck.get('shared'):
        print('TEAM BALANCE  --  the same network plays both teams, with the')
        print('starting seat alternated so turn order cannot be mistaken for')
        print('team strength.\n')
        r = team_balance(netA, args.games * 4)
        lo, hi = wilson(r['mine_overall'] / 100, r['n'])
        print('  Bonzumi + Sipzap win rate   %.1f%%   (95%% CI %.1f-%.1f, n=%d)'
              % (r['mine_overall'], 100 * lo, 100 * hi, r['n']))
        print('      moving first             %.1f%%' % r['mine_first'])
        print('      moving second            %.1f%%' % r['mine_second'])
        print('      first-move advantage     %+.1f points' % r['first_move_edge'])
        print('  above 50% means Bonzumi + Sipzap is the stronger roster.')
        print('  the hand-tuned solver puts this at 46.0%.\n')

    print('WHERE THE GAMES ARE BEING LOST')
    print('the network in seat 0 against the hand-tuned policy in seat 1,')
    print('both playing Bonzumi + Sipzap and Pelijet + Barbenin respectively.\n')
    beh, wr, ng, honey = behaviour(netA, hB, 300)
    print('  %-18s %12s %12s' % ('per game', 'network', 'heuristic'))
    for key, label in (('damage', 'damage dealt'), ('fires', 'strikes'),
                       ('mana', 'mana earned'), ('wasted_mana', 'mana burnt'),
                       ('berries', 'berries taken'), ('denied', 'DENIAL tiles'),
                       ('evolves', 'EVOLUTIONS'),
                       ('boosts', 'boosts')):
        print('  %-18s %12.2f %12.2f' % (label, beh[0][key], beh[1][key]))
    print('  %-18s %12.1f %12s' % ('turns per game', beh[0]['turns'], '(shared)'))
    usable0 = beh[0]['mana'] - beh[0]['wasted_mana']
    usable1 = beh[1]['mana'] - beh[1]['wasted_mana']
    print('  %-18s %12.2f %12.2f' % ('usable mana', usable0, usable1))
    print('  %-18s %12.2f %12.2f' % ('damage per mana',
                                     beh[0]['damage'] / max(usable0, 1e-9),
                                     beh[1]['damage'] / max(usable1, 1e-9)))
    print('  %-18s %12.2f %12.2f' % ('damage per strike',
                                     beh[0]['damage'] / max(beh[0]['fires'], 1e-9),
                                     beh[1]['damage'] / max(beh[1]['fires'], 1e-9)))
    tot = sum(honey[0].values())
    if tot:
        print('\n  Sipzap / Ranzap fired at honey level (network, seat 0):')
        print('  %-10s %10s %10s %10s' % ('honey', 'share', 'damage', 'wasted'))
        for h in sorted(honey[0]):
            dmg = [0, 5, 10, 20][min(h, 3)]
            print('  %-10d %9.0f%% %10d %10s'
                  % (h, 100 * honey[0][h] / tot, dmg,
                     'yes' if dmg == 0 else ''))
        print('  every strike at honey 0 throws 4 mana away for no damage.')
    print('  network win rate in this matchup: %.1f%% over %d games\n' % (wr, ng))

    rows = []
    if args.search:
        srch = SearchAgent(A, device, samples=args.search, top_k=args.search_top)
        rows += [
            ('SEARCH (net as evaluator)  vs  heuristic pel_deny', srch, hB),
            ('heuristic bon_hdeny        vs  SEARCH', hA,
             SearchAgent(B, device, samples=args.search, top_k=args.search_top)),
        ]
    rows += [
        ('net A (Bonzumi+Sipzap)  vs  net B (Pelijet+Barbenin)', netA, netB),
        ('net A                   vs  heuristic pel_deny', netA, hB),
        ('heuristic bon_hdeny     vs  net B', hA, netB),
        ('heuristic bon_hdeny     vs  heuristic pel_deny', hA, hB),
    ]
    print('%-52s %8s %16s' % ('matchup', 'side-0 %', '95% CI'))
    for label, a0, a1 in rows:
        p, n = play(a0, a1, args.games)
        lo, hi = wilson(p, n)
        print('%-52s %7.1f%% %16s' % (label, 100 * p,
                                      '%.1f-%.1f' % (100 * lo, 100 * hi)))
    print('\nside-0 is always Bonzumi + Sipzap.')
    print('net A beating the heuristic means the learned agent found something '
          'the hand-tuned scorer did not.')


if __name__ == '__main__':
    main()
