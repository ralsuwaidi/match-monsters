"""Evaluate the co-evolved networks: against each other, and against the
hand-tuned heuristics, so you can see which side actually got better.

    uv run --extra rl python eval_nn.py --games 400
"""
import argparse
import random

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


def play(agent0, agent1, n_games, seed=0, batch=128, max_turns=60):
    teams = (engine.MY_TEAM, engine.FOE_TEAM)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='checkpoints/selfplay.pt')
    ap.add_argument('--games', type=int, default=400)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--sample', action='store_true',
                    help='sample from the policy instead of taking the argmax')
    args = ap.parse_args()

    device = nets.pick_device(args.device)
    ck = torch.load(args.ckpt, map_location=device)
    arch = ck.get('arch', {})
    A = nets.ActorCritic(**arch).to(device).eval()
    B = nets.ActorCritic(**arch).to(device).eval()
    if ck.get('shared'):
        A.load_state_dict(ck['net']); B.load_state_dict(ck['net'])
        print('shared checkpoint: the same network plays both teams')
    else:
        A.load_state_dict(ck['A']); B.load_state_dict(ck['B'])
    print(f'checkpoint from iteration {ck["iter"]}, device {device}\n')

    netA = NetAgent(A, device, greedy=not args.sample)
    netB = NetAgent(B, device, greedy=not args.sample)
    hA = HeuristicAgent(ai.MY_POLICIES['bon_hdeny'])
    hB = HeuristicAgent(ai.FOE_POLICIES['pel_deny'])

    rows = [
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
