"""One network, playing itself, learning the game from scratch.

A single shared policy plays BOTH teams. That is possible because the
observation is side-relative and describes each monster by its stats -- cost,
damage, whether it charges, what its ability does -- rather than by name. The
network never learns "red is good"; it learns "a bar I can fill cheaply is
worth filling", and applies that to whichever monsters it happens to have.

Three things this buys over two separate networks:
  * every game produces training data from both sides, so twice the data
  * no arms race between two parameter sets learning at different speeds
  * the win rate becomes a clean measurement -- when the SAME brain plays both
    teams, any deviation from 50% is the ROSTERS, not the training

    uv run --extra rl python train_selfplay.py --iters 1500
"""
import argparse
import os
import random
import signal
import time

import numpy as np
import torch
import torch.nn.functional as F

from match_monsters.agents import ai
from match_monsters.game import engine
from match_monsters.rl.selfplay import N_SWAP
from match_monsters.game import grid
from match_monsters.rl import nets
from match_monsters.solver import progress
from match_monsters.rl.selfplay import Duel

_STOP = False


def _on_term(sig, frm):
    global _STOP
    _STOP = True


class Buffer:
    def __init__(self):
        self.b, self.s, self.m, self.a = [], [], [], []
        self.lp, self.v, self.r, self.done, self.ep = [], [], [], [], []

    def add(self, b, s, m, a, lp, v, ep):
        self.b.append(b); self.s.append(s); self.m.append(m); self.a.append(a)
        self.lp.append(lp); self.v.append(v); self.ep.append(ep)
        self.r.append(0.0); self.done.append(0.0)
        return len(self.a) - 1

    def __len__(self):
        return len(self.a)


def gae(buf, gamma, lam):
    n = len(buf)
    adv = np.zeros(n, np.float32); ret = np.zeros(n, np.float32)
    v = np.asarray(buf.v, np.float32)
    r = np.asarray(buf.r, np.float32)
    dn = np.asarray(buf.done, np.float32)
    i = n - 1
    while i >= 0:
        ep = buf.ep[i]
        j = i
        while j >= 0 and buf.ep[j] == ep:
            j -= 1
        g_ = 0.0
        nxt = 0.0 if dn[i] else v[i]     # bootstrap unfinished episodes
        for k in range(i, j, -1):
            d = dn[k]
            delta = r[k] + gamma * nxt * (1 - d) - v[k]
            g_ = delta + gamma * lam * (1 - d) * g_
            adv[k] = g_; ret[k] = g_ + v[k]; nxt = v[k]
        i = j
    return ret, adv


def ppo_update(net, opt, buf, ret, adv, device, epochs, batch, clip, vf, ent):
    b = torch.as_tensor(np.asarray(buf.b), device=device)
    s = torch.as_tensor(np.asarray(buf.s), device=device)
    m = torch.as_tensor(np.asarray(buf.m), device=device)
    a = torch.as_tensor(np.asarray(buf.a), dtype=torch.long, device=device)
    old = torch.as_tensor(np.asarray(buf.lp, np.float32), device=device)
    rt = torch.as_tensor(ret, device=device)
    ad = torch.as_tensor(adv, device=device)
    ad = (ad - ad.mean()) / (ad.std() + 1e-8)
    idx = np.arange(len(a)); out = {}
    for _ in range(epochs):
        np.random.shuffle(idx)
        for k in range(0, len(idx), batch):
            j = torch.as_tensor(idx[k:k + batch], dtype=torch.long, device=device)
            logits, value = net(b[j], s[j], m[j])
            dist = torch.distributions.Categorical(logits=logits)
            lp = dist.log_prob(a[j])
            ratio = (lp - old[j]).exp()
            pl = -torch.min(ratio * ad[j],
                            torch.clamp(ratio, 1 - clip, 1 + clip) * ad[j]).mean()
            vl = F.mse_loss(value, rt[j])
            e = dist.entropy().mean()
            opt.zero_grad(set_to_none=True)
            (pl + vf * vl - ent * e).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            out = {'policy_loss': pl.detach().item(),
                   'value_loss': vl.detach().item(),
                   'entropy': e.detach().item()}
    return out


@torch.no_grad()
def rollout(net, duels, rngs, device, steps, teams, max_turns, shaping,
            match_only_p=0.0, waste_penalty=0.0):
    """Every live game steps together in ONE batched forward pass -- possible
    because the observation is side-relative, so the same net answers for
    whichever side happens to be on move."""
    buf = Buffer()
    results = []
    tele = []            # one row per completed game
    last_ix, last_diff, ep_no = {}, {}, [0] * len(duels)

    def credit(i, side, diff, terminal=None):
        key = (i, side)
        if key in last_ix:
            buf.r[last_ix[key]] += shaping * (diff - last_diff[key])
            if terminal is not None:
                buf.r[last_ix[key]] += terminal
                buf.done[last_ix[key]] = 1.0
                del last_ix[key]
                return
        last_diff[key] = diff

    while len(buf) < steps:
        live = [i for i, d in enumerate(duels) if not d.done]
        obs = [duels[i].observe() for i in live]
        mo = np.random.rand(len(live)) < match_only_p
        masks = np.asarray([duels[i].legal_mask(bool(mo[k]))
                            for k, i in enumerate(live)])
        boards = np.asarray([o[0] for o in obs])
        scals = np.asarray([o[1] for o in obs])
        logits, vals = net(torch.as_tensor(boards, device=device),
                           torch.as_tensor(scals, device=device),
                           torch.as_tensor(masks, device=device))
        dist = torch.distributions.Categorical(logits=logits)
        acts = dist.sample()
        lps = dist.log_prob(acts).cpu().numpy()
        acts = acts.cpu().numpy(); vals = vals.cpu().numpy()
        for k, i in enumerate(live):
            d = duels[i]
            side = d.active
            diff = (d.sides[side].hp - d.sides[1 - side].hp) / rules_base()
            credit(i, side, diff)
            ix = buf.add(boards[k], scals[k], masks[k], int(acts[k]),
                         float(lps[k]), float(vals[k]), (i, side, ep_no[i]))
            last_ix[(i, side)] = ix
            d.step(int(acts[k]))
            # A move that clears nothing is a wasted move. Penalising it
            # directly gives the losing actions GRADIENT, which masking them
            # does not -- a masked logit is never sampled, never updated, and
            # springs back the moment the mask is lifted. Annealed to zero, so
            # the final policy can still learn that repositioning is sometimes
            # the right move.
            if waste_penalty and not d.last_matched and int(acts[k]) < N_SWAP:
                buf.r[ix] -= waste_penalty
            if d.done:
                results.append(d.result)
                st = d.st
                dmg = sum(v for k, v in st.items() if '/dmg_' in k)
                fires = sum(v for k, v in st.items() if '/fires_' in k)
                waste = sum(v for k, v in st.items() if '/wasted_' in k)
                mana = sum(v for k, v in st.items() if '/mana_' in k)
                tele.append((d.turn, d.n_swaps, d.n_matched, d.n_big,
                             dmg, fires, waste, mana,
                             1.0 if d.result in (0.0, 1.0) else 0.0))
                for sd in (0, 1):
                    term = (2 * d.result - 1) if sd == 0 else (1 - 2 * d.result)
                    dd = (d.sides[sd].hp - d.sides[1 - sd].hp) / rules_base()
                    credit(i, sd, dd, terminal=term)
                last_ix.pop((i, 0), None); last_ix.pop((i, 1), None)
                last_diff.pop((i, 0), None); last_diff.pop((i, 1), None)
                ep_no[i] += 1
                rngs[i] = random.Random(rngs[i].randrange(1 << 30))
                duels[i] = Duel(rngs[i], teams, max_turns=max_turns)
    return buf, results, tele


def rules_base():
    from match_monsters import rules
    return float(rules.BASE_HP)


@torch.no_grad()
def eval_vs_heuristics(net, device, games, max_turns):
    """The absolute yardstick: the shared net plays BOTH teams against the
    hand-tuned policies. Self-play win rate alone cannot tell you whether the
    agent is good, only whether it is balanced against itself."""
    from match_monsters.rl import evaluate as eval_nn
    net.eval()
    agent = eval_nn.NetAgent(net, device, greedy=False)
    hA = eval_nn.HeuristicAgent(ai.MY_POLICIES['bon_hdeny'])
    hB = eval_nn.HeuristicAgent(ai.FOE_POLICIES['pel_deny'])
    as_a, _ = eval_nn.play(agent, hB, games, seed=11, max_turns=max_turns)
    as_b, _ = eval_nn.play(hA, agent, games, seed=11, max_turns=max_turns)
    net.train()
    return 100 * as_a, 100 * (1 - as_b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', default=None)
    ap.add_argument('--iters', type=int, default=1500)
    ap.add_argument('--games', type=int, default=192)
    ap.add_argument('--steps', type=int, default=8192)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--gamma', type=float, default=0.997)
    ap.add_argument('--lam', type=float, default=0.95)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--clip', type=float, default=0.2)
    ap.add_argument('--vf', type=float, default=0.5)
    ap.add_argument('--entropy', type=float, default=0.02)
    ap.add_argument('--entropy-final', type=float, default=0.003)
    ap.add_argument('--shaping', type=float, default=0.15)
    ap.add_argument('--shaping-anneal', type=float, default=0.6)
    ap.add_argument('--waste-penalty', type=float, default=0.15,
                    help='reward subtracted for a move that clears nothing, '
                         'annealed away over --match-curriculum')
    ap.add_argument('--match-curriculum', type=float, default=0.6,
                    help='fraction of training over which the "matches only" '
                         'restriction is annealed from 1.0 to 0. Set 0 to '
                         'learn from nothing (which does not work -- see '
                         'the diagnostic in the README).')
    ap.add_argument('--width', type=int, default=96)
    ap.add_argument('--blocks', type=int, default=4)
    ap.add_argument('--hidden', type=int, default=384)
    ap.add_argument('--eval-every', type=int, default=25)
    ap.add_argument('--eval-games', type=int, default=200)
    ap.add_argument('--max-turns', type=int, default=60)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--ckpt', default='checkpoints')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)
    device = nets.pick_device(args.device)
    os.makedirs(args.ckpt, exist_ok=True)
    arch = dict(width=args.width, blocks=args.blocks, hidden=args.hidden)
    net = nets.ActorCritic(**arch).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    path = os.path.join(args.ckpt, 'selfplay.pt')
    start = 1
    if args.resume and os.path.exists(path):
        ck = torch.load(path, map_location=device)
        if ck.get('arch') and ck['arch'] != arch:
            raise SystemExit(f'checkpoint arch {ck["arch"]} != requested {arch}')
        net.load_state_dict(ck['net']); opt.load_state_dict(ck['opt'])
        start = ck['iter'] + 1
        print(f'resumed at iteration {ck["iter"]}')

    run_id = args.run_id or ('sp-' + progress.new_run_id())
    teams = (engine.MY_TEAM, engine.FOE_TEAM)
    rngs = [random.Random(500 + i) for i in range(args.games)]
    duels = [Duel(rngs[i], teams, max_turns=args.max_turns)
             for i in range(args.games)]
    nparams = sum(p.numel() for p in net.parameters())
    state = {'run_id': run_id, 'kind': 'neural', 'pid': os.getpid(),
             'started': time.time(), 'status': 'running', 'iter': 0,
             'iters_planned': args.iters, 'device': str(device),
             'shared': True,
             'config': {'games': args.games, 'steps': args.steps, 'lr': args.lr,
                        'entropy': args.entropy, 'gamma': args.gamma,
                        'berry_rate': grid.BERRY_WEIGHT,
                        'board': f'{grid.W}x{grid.H}', 'arch': arch,
                        'shaping': args.shaping, 'params': nparams},
             'history': [], 'error': None}
    progress.write(run_id, state)
    # a self-contained log next to the run, so a Colab session can be pasted
    # back for review without hunting for the notebook output
    os.makedirs(progress.run_dir(run_id), exist_ok=True)
    logf = open(os.path.join(progress.run_dir(run_id), 'train.log'), 'a',
                buffering=1)

    def say(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')

    say(f'run {run_id}  device {device}  {nparams:,} params  '
        f'(ONE net playing both sides)')
    say(f'arch {arch}  games {args.games}  steps/iter {args.steps}  '
        f'lr {args.lr}  entropy {args.entropy}->{args.entropy_final}  '
        f'shaping {args.shaping}  waste_penalty {args.waste_penalty}')

    total, recent, t0 = 0, [], time.time()
    for it in range(start, args.iters + 1):
        if _STOP:
            break
        shaping = args.shaping * max(0.0, 1.0 - it / max(1, args.iters * args.shaping_anneal))
        ent = args.entropy + (args.entropy_final - args.entropy) * min(
            1.0, it / max(1, args.iters * 0.7))
        prog = min(1.0, it / max(1e-9, args.iters * args.match_curriculum))
        mo_p = 0.0                      # masking is kept only for comparison
        wp = args.waste_penalty * (1.0 - prog)
        buf, results, tele = rollout(net, duels, rngs, device, args.steps, teams,
                                     args.max_turns, shaping, mo_p, wp)
        ret, adv = gae(buf, args.gamma, args.lam)
        stats = ppo_update(net, opt, buf, ret, adv, device, args.epochs,
                           args.batch, args.clip, args.vf, ent)
        total += len(buf)
        recent = (recent + results)[-4000:]
        wr = 100 * float(np.mean(recent)) if recent else float('nan')
        T = np.asarray(tele, np.float64) if tele else np.zeros((0, 9))
        tel = {}
        if len(T):
            tel = {
                'turns_to_win': float(T[T[:, 8] == 1, 0].mean()) if (T[:, 8] == 1).any() else None,
                'match_rate': 100 * float(T[:, 2].sum() / max(1, T[:, 1].sum())),
                'big_rate': 100 * float(T[:, 3].sum() / max(1, T[:, 2].sum())),
                'damage': float(T[:, 4].mean()),
                'fires': float(T[:, 5].mean()),
                'wasted_mana': float(T[:, 6].mean()),
                'mana': float(T[:, 7].mean()),
                'decisive': 100 * float(T[:, 8].mean()),
            }
        ev_a = ev_b = None
        if args.eval_every and (it % args.eval_every == 0 or it == 1):
            ev_a, ev_b = eval_vs_heuristics(net, device, args.eval_games,
                                            args.max_turns)
        el = time.time() - t0
        state['iter'] = it
        state['elapsed'] = el
        state['history'].append({
            'iter': it, 'wr': wr, 'wr_rolling': wr, 'games': len(results),
            'steps': total, 'sps': total / max(el, 1e-9),
            'entropy_A': stats['entropy'], 'entropy_B': stats['entropy'],
            'vloss_A': stats['value_loss'], 'vloss_B': stats['value_loss'],
            'shaping': shaping, 'entropy_coef': ent, 'waste_penalty': wp,
            'A_vs_heuristic': ev_a, 'B_vs_heuristic': ev_b, **tel})
        progress.write(run_id, state)
        ev_txt = '' if ev_a is None else f'  | vs heuristic  as-A {ev_a:.0f}%  as-B {ev_b:.0f}%'
        say(f'iter {it:4d}  win {wr:5.1f}%  '
              f'turns {tel.get("turns_to_win") or float("nan"):5.1f}  '
              f'match {tel.get("match_rate", 0):4.1f}%  '
              f'dmg {tel.get("damage", 0):5.1f}  '
              f'fires {tel.get("fires", 0):4.1f}  '
              f'decisive {tel.get("decisive", 0):3.0f}%  '
              f'wp {wp:.3f}  '
            f'{total/max(el,1e-9):.0f}/s  ent {stats["entropy"]:.2f}{ev_txt}')
        torch.save({'net': net.state_dict(), 'opt': opt.state_dict(),
                    'arch': arch, 'iter': it, 'shared': True}, path)

    state['status'] = 'stopped' if _STOP else 'done'
    progress.write(run_id, state)


if __name__ == '__main__':
    main()
