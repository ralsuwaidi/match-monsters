"""Co-evolutionary self-play: two neural agents learning the game from scratch.

Neither network is given a rule, a heuristic, or a hint about what a tile does.
Each sees the raw board, its own and its opponent's bars, and which actions are
legal. Everything else is learned from winning and losing.

Both sides train at once against each other, so the opponent gets harder exactly
as fast as the agent does. The head-to-head win rate is logged every iteration --
if it settles, that number is a property of the two ROSTERS rather than of
anyone's tuning.

    uv run --extra rl python train_coevolve.py --iters 200

Laptop and Colab run the same command; the device is picked automatically.
The engine is pure Python and is the bottleneck, so a GPU speeds the updates
but not the rollouts -- expect a modest gain from CUDA, a large one from cores.
"""
import argparse
import os
import random
import signal
import time

import numpy as np
import torch
import torch.nn.functional as F

import engine
import nets
import progress
import selfplay
import ai
from selfplay import Duel, N_ACTIONS, SCALARS, TILES
import grid

_STOP = False


def _on_term(sig, frm):
    global _STOP
    _STOP = True


class Buffer:
    """Transitions for one side. Rewards are filled in late: the consequence of
    a move includes what the opponent does before this side moves again."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.b, self.s, self.m, self.a = [], [], [], []
        self.lp, self.v, self.ep_id = [], [], []
        self.r, self.done = [], []

    def add(self, b, s, m, a, lp, v, ep):
        self.b.append(b); self.s.append(s); self.m.append(m)
        self.a.append(a); self.lp.append(lp); self.v.append(v); self.ep_id.append(ep)
        self.r.append(0.0); self.done.append(0.0)
        return len(self.a) - 1

    def __len__(self):
        return len(self.a)


def returns_and_adv(buf, gamma, lam):
    """GAE over each side's own decision sequence.

    An episode still running when the rollout ends is BOOTSTRAPPED from the
    value head, not treated as terminal -- otherwise every unfinished game
    teaches the critic that the position was worth zero.
    """
    n = len(buf)
    adv = np.zeros(n, np.float32)
    ret = np.zeros(n, np.float32)
    v = np.asarray(buf.v, np.float32)
    r = np.asarray(buf.r, np.float32)
    dn = np.asarray(buf.done, np.float32)
    i = n - 1
    while i >= 0:
        ep = buf.ep_id[i]
        j = i
        while j >= 0 and buf.ep_id[j] == ep:
            j -= 1
        gae = 0.0
        nxt_v = 0.0 if dn[i] else v[i]      # bootstrap if it did not finish
        for k in range(i, j, -1):
            done = dn[k]
            delta = r[k] + gamma * nxt_v * (1 - done) - v[k]
            gae = delta + gamma * lam * (1 - done) * gae
            adv[k] = gae
            ret[k] = gae + v[k]
            nxt_v = v[k]
        i = j
    return ret, adv


def ppo_update(net, opt, buf, ret, adv, device, epochs=4, batch=512,
               clip=0.2, vf=0.5, ent=0.01):
    b = torch.as_tensor(np.asarray(buf.b), device=device)
    s = torch.as_tensor(np.asarray(buf.s), device=device)
    m = torch.as_tensor(np.asarray(buf.m), device=device)
    a = torch.as_tensor(np.asarray(buf.a), dtype=torch.long, device=device)
    old = torch.as_tensor(np.asarray(buf.lp, np.float32), device=device)
    ret_t = torch.as_tensor(ret, device=device)
    adv_t = torch.as_tensor(adv, device=device)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    n = len(a)
    idx = np.arange(n)
    stats = {}
    for _ in range(epochs):
        np.random.shuffle(idx)
        for k in range(0, n, batch):
            j = torch.as_tensor(idx[k:k + batch], dtype=torch.long, device=device)
            logits, value = net(b[j], s[j], m[j])
            dist = torch.distributions.Categorical(logits=logits)
            lp = dist.log_prob(a[j])
            ratio = (lp - old[j]).exp()
            l1 = ratio * adv_t[j]
            l2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[j]
            pl = -torch.min(l1, l2).mean()
            vl = F.mse_loss(value, ret_t[j])
            e = dist.entropy().mean()
            loss = pl + vf * vl - ent * e
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            stats = {'policy_loss': pl.detach().item(),
                     'value_loss': vl.detach().item(),
                     'entropy': e.detach().item()}
    return stats


@torch.no_grad()
def act(net, boards, scalars, masks, device, greedy=False):
    b = torch.as_tensor(np.asarray(boards), device=device)
    s = torch.as_tensor(np.asarray(scalars), device=device)
    m = torch.as_tensor(np.asarray(masks), device=device)
    logits, v = net(b, s, m)
    if greedy:
        a = logits.argmax(-1)
        return a.cpu().numpy(), None, v.cpu().numpy()
    dist = torch.distributions.Categorical(logits=logits)
    a = dist.sample()
    return a.cpu().numpy(), dist.log_prob(a).cpu().numpy(), v.cpu().numpy()


def rollout(duels, rngs, netA, netB, device, steps, bufs,
            ep_counter, results, teams, max_turns, shaping):
    """Step every live game until each side has `steps` transitions.

    A move's reward is only known once the opponent has replied, so each
    transition is credited when that side is next on move (or when the game
    ends): the change in HP difference over that whole interval.
    """
    last_ix = {}      # (duel, side) -> index of that side's previous transition
    last_diff = {}

    def credit(i, side, hp_diff, terminal=None):
        key = (i, side)
        if key in last_ix:
            ix = last_ix[key]
            bufs[side].r[ix] += shaping * (hp_diff - last_diff[key])
            if terminal is not None:
                bufs[side].r[ix] += terminal
                bufs[side].done[ix] = 1.0
                del last_ix[key]
                return
        last_diff[key] = hp_diff

    while len(bufs[0]) < steps or len(bufs[1]) < steps:
        for side in (0, 1):
            live = [i for i, d in enumerate(duels)
                    if not d.done and d.active == side]
            if not live:
                continue
            obs = [duels[i].observe() for i in live]
            masks = [duels[i].legal_mask() for i in live]
            boards = [o[0] for o in obs]
            scals = [o[1] for o in obs]
            net = netA if side == 0 else netB
            acts, lps, vals = act(net, boards, scals, masks, device)
            for k, i in enumerate(live):
                d = duels[i]
                diff = (d.sides[side].hp - d.sides[1 - side].hp) / 80.0
                credit(i, side, diff)
                ix = bufs[side].add(boards[k], scals[k], masks[k], int(acts[k]),
                                    float(lps[k]), float(vals[k]),
                                    (i, ep_counter[i]))
                last_ix[(i, side)] = ix
                d.step(int(acts[k]))
                if d.done:
                    r0 = d.result
                    results.append(r0)
                    for sd in (0, 1):
                        term = (2 * r0 - 1) if sd == 0 else (1 - 2 * r0)
                        dd = (d.sides[sd].hp - d.sides[1 - sd].hp) / 80.0
                        credit(i, sd, dd, terminal=term)
                    last_ix.pop((i, 0), None); last_ix.pop((i, 1), None)
                    last_diff.pop((i, 0), None); last_diff.pop((i, 1), None)
                    ep_counter[i] += 1
                    rngs[i] = random.Random(rngs[i].randrange(1 << 30))
                    duels[i] = Duel(rngs[i], teams, max_turns=max_turns)


@torch.no_grad()
def eval_vs_heuristic(netA, netB, device, games, max_turns):
    """Absolute yardstick.

    The self-play win rate is RELATIVE -- it stays near 50% whether both agents
    are brilliant or both are useless. Playing each net against a fixed
    hand-tuned policy shows whether they are actually getting better.
    """
    import eval_nn
    netA.eval(); netB.eval()
    a = eval_nn.NetAgent(netA, device, greedy=False)
    b = eval_nn.NetAgent(netB, device, greedy=False)
    hA = eval_nn.HeuristicAgent(ai.MY_POLICIES['bon_hdeny'])
    hB = eval_nn.HeuristicAgent(ai.FOE_POLICIES['pel_deny'])
    a_vs_h, _ = eval_nn.play(a, hB, games, seed=7, max_turns=max_turns)
    h_vs_b, _ = eval_nn.play(hA, b, games, seed=7, max_turns=max_turns)
    netA.train(); netB.train()
    # a_vs_h: net A's win rate.  h_vs_b: heuristic's win rate, so net B's is 1-x
    return 100 * a_vs_h, 100 * (1 - h_vs_b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', default=None)
    ap.add_argument('--iters', type=int, default=300)
    ap.add_argument('--games', type=int, default=128, help='parallel games')
    ap.add_argument('--steps', type=int, default=6144, help='transitions/side/iter')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--gamma', type=float, default=0.997)
    ap.add_argument('--lam', type=float, default=0.95)
    ap.add_argument('--entropy', type=float, default=0.02)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--ckpt', default='checkpoints')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--max-turns', type=int, default=60)
    ap.add_argument('--shaping', type=float, default=0.15,
                    help='dense reward per unit of HP-difference swing; gives '
                         'the agents a gradient before they can win on purpose')
    ap.add_argument('--shaping-anneal', type=float, default=0.6,
                    help='fraction of the run over which shaping decays to 0. '
                         'Dense HP reward favours the team that hits often and '
                         'small (Pelijet) over the one that hits rarely and big '
                         '(Bonzumi), so leaving it on biases the result.')
    ap.add_argument('--eval-every', type=int, default=25,
                    help='iterations between absolute evaluations (0 to skip)')
    ap.add_argument('--eval-games', type=int, default=200)
    ap.add_argument('--entropy-final', type=float, default=0.003,
                    help='entropy bonus decays to this, so the agents commit '
                         'to a strategy instead of staying near-random')
    ap.add_argument('--league', type=int, default=1,
                    help='1 = alternate which side learns, against a pool of '
                         'the other side\'s past snapshots. Stops the two nets '
                         'chasing each other and diverging.')
    ap.add_argument('--snap-every', type=int, default=20)
    ap.add_argument('--pool', type=int, default=5)
    ap.add_argument('--width', type=int, default=128)
    ap.add_argument('--blocks', type=int, default=6)
    ap.add_argument('--hidden', type=int, default=512)
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    device = nets.pick_device(args.device)
    os.makedirs(args.ckpt, exist_ok=True)
    arch = dict(width=args.width, blocks=args.blocks, hidden=args.hidden)
    netA = nets.ActorCritic(**arch).to(device)     # Bonzumi + Sipzap
    netB = nets.ActorCritic(**arch).to(device)     # Pelijet + Barbenin
    optA = torch.optim.Adam(netA.parameters(), lr=args.lr)
    optB = torch.optim.Adam(netB.parameters(), lr=args.lr)
    start_iter = 1
    ckpt_path = os.path.join(args.ckpt, 'coevolve.pt')
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        if ck.get('arch') and ck['arch'] != arch:
            raise SystemExit(f'checkpoint architecture {ck["arch"]} does not '
                             f'match requested {arch}')
        netA.load_state_dict(ck['A']); netB.load_state_dict(ck['B'])
        optA.load_state_dict(ck['optA']); optB.load_state_dict(ck['optB'])
        start_iter = ck['iter'] + 1
        print(f'resumed from iteration {ck["iter"]}')

    run_id = args.run_id or ('nn-' + progress.new_run_id())
    teams = (engine.MY_TEAM, engine.FOE_TEAM)
    rngs = [random.Random(1000 + i) for i in range(args.games)]
    duels = [Duel(rngs[i], teams, max_turns=args.max_turns)
             for i in range(args.games)]
    ep_counter = [0] * args.games

    state = {'run_id': run_id, 'kind': 'neural', 'pid': os.getpid(),
             'started': time.time(), 'status': 'running', 'iter': 0,
             'iters_planned': args.iters, 'device': str(device),
             'config': {'games': args.games, 'steps': args.steps, 'lr': args.lr,
                        'entropy': args.entropy, 'gamma': args.gamma,
                        'berry_rate': grid.BERRY_WEIGHT,
                        'board': f'{grid.W}x{grid.H}',
                        'shaping': args.shaping,
                        'shaping_anneal': args.shaping_anneal,
                        'arch': arch,
                        'params': sum(p.numel() for p in netA.parameters())},
             'history': [], 'error': None}
    progress.write(run_id, state)
    print(f'run {run_id}  device {device}  '
          f'{sum(p.numel() for p in netA.parameters()):,} params/side', flush=True)

    total_steps = 0
    recent = []
    poolA, poolB = [], []          # past snapshots, kept on cpu
    oppA = nets.ActorCritic(**arch).to(device).eval()
    oppB = nets.ActorCritic(**arch).to(device).eval()
    t0 = time.time()
    for it in range(start_iter, args.iters + 1):
        if _STOP:
            break
        # decay the dense reward away so the endgame is trained on the real
        # objective -- winning -- not on the shaping proxy
        frac = it / max(1, args.iters * args.shaping_anneal)
        shaping = args.shaping * max(0.0, 1.0 - frac)
        # and decay the entropy bonus so the agents actually commit
        ent_coef = args.entropy + (args.entropy_final - args.entropy) * min(
            1.0, it / max(1, args.iters * 0.7))

        # League: only one side learns per iteration, and it plays a snapshot
        # of the other side rather than the other side's live weights. Two nets
        # both chasing each other's current policy is what makes co-evolution
        # oscillate instead of settle.
        if args.league:
            learner = 0 if (it % 2 == 1) else 1
            pool = poolB if learner == 0 else poolA
            opp = oppB if learner == 0 else oppA
            live = netB if learner == 0 else netA
            opp.load_state_dict(random.choice(pool) if pool else live.state_dict())
            useA = netA if learner == 0 else opp
            useB = opp if learner == 0 else netB
        else:
            learner = None
            useA, useB = netA, netB

        bufs = [Buffer(), Buffer()]
        results = []
        rollout(duels, rngs, useA, useB, device, args.steps, bufs,
                ep_counter, results, teams, args.max_turns, shaping)

        stats = {0: {}, 1: {}}
        sides = ((netA, optA, 0), (netB, optB, 1)) if learner is None else (
            ((netA, optA, 0),) if learner == 0 else ((netB, optB, 1),))
        for net, opt, side in sides:
            ret, adv = returns_and_adv(bufs[side], args.gamma, args.lam)
            stats[side] = ppo_update(net, opt, bufs[side], ret, adv, device,
                                     ent=ent_coef)

        if args.league and it % args.snap_every == 0:
            poolA.append({k: v.detach().cpu().clone()
                          for k, v in netA.state_dict().items()})
            poolB.append({k: v.detach().cpu().clone()
                          for k, v in netB.state_dict().items()})
            poolA[:] = poolA[-args.pool:]
            poolB[:] = poolB[-args.pool:]
        total_steps += len(bufs[0]) + len(bufs[1])
        ev_a = ev_b = None
        if args.eval_every and (it % args.eval_every == 0 or it == 1):
            ev_a, ev_b = eval_vs_heuristic(netA, netB, device,
                                           args.eval_games, args.max_turns)

        recent.extend(results)
        recent = recent[-4000:]
        wr = float(np.mean(results)) if results else float('nan')
        wr_roll = float(np.mean(recent)) if recent else float('nan')
        el = time.time() - t0
        state['iter'] = it
        state['elapsed'] = el
        state['history'].append({
            'iter': it, 'wr': 100 * wr, 'wr_rolling': 100 * wr_roll,
            'shaping': shaping, 'entropy_coef': ent_coef,
            'learner': learner, 'games': len(results),
            'A_vs_heuristic': ev_a, 'B_vs_heuristic': ev_b,
            'steps': total_steps, 'sps': total_steps / max(el, 1e-9),
            'entropy_A': stats[0].get('entropy'), 'entropy_B': stats[1].get('entropy'),
            'vloss_A': stats[0].get('value_loss'), 'vloss_B': stats[1].get('value_loss'),
        })
        progress.write(run_id, state)
        ev_txt = ('' if ev_a is None else
                  f'  | vs heuristic  A {ev_a:.0f}%  B {ev_b:.0f}%')
        print(f'iter {it:4d}  side-0 win {100*wr_roll:5.1f}%  games {len(results):4d}  '
              f'steps {total_steps:,}  {total_steps/max(el,1e-9):.0f}/s  '
              f'ent {stats[0].get("entropy", 0):.2f}/{stats[1].get("entropy", 0):.2f}'
              f'{ev_txt}', flush=True)
        torch.save({'A': netA.state_dict(), 'B': netB.state_dict(),
                    'optA': optA.state_dict(), 'optB': optB.state_dict(),
                    'arch': arch, 'iter': it}, ckpt_path)

    state['status'] = 'stopped' if _STOP else 'done'
    progress.write(run_id, state)


if __name__ == '__main__':
    main()
