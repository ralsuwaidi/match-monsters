"""Behaviour cloning: learn to imitate the hand-tuned policies, then let RL
improve from there.

Every run so far started from random weights and tried to *discover* competent
play by exploration -- which is exactly what kept failing. This skips that
entirely. A strong player already exists, so the network first learns to
predict what it would do (supervised, cross-entropy on its chosen action), and
picks up a value head trained on the actual game outcomes at the same time.

RL then starts from a competent policy and improves it, instead of spending
hundreds of iterations rediscovering that matching tiles is a good idea.

    mm-pretrain --games 8000            collect and train
    mm-train --resume                   continue with PPO from the result
"""
import argparse
import os
import random
import time
from multiprocessing import Pool

import numpy as np

from match_monsters.agents import ai
from match_monsters.game import engine
from match_monsters.rl.selfplay import Duel, N_ACTIONS, SWAP_IX

# Draw from several policies so the network learns the shared idea of good play
# rather than memorising one style's quirks.
# Only strong policies -- cloning weak play teaches weak play. These are the
# top three per team by the solver's floor, kept plural so the network learns
# the shared idea of good play rather than one style's quirks.
MINE = ['bon_hdeny', 'bon_deny', 'bon_berry2']
THEIRS = ['pel_deny', 'pel_hdeny', 'self_first']


def _agent(policy):
    from match_monsters.rl import evaluate
    return evaluate.HeuristicAgent(policy)


def collect_shard(args):
    """Play games between hand-tuned policies, recording every decision."""
    n_games, seed, max_turns = args
    rng = random.Random(seed)
    teams = (engine.MY_TEAM, engine.FOE_TEAM)
    B, S, M, A, W = [], [], [], [], []
    for g in range(n_games):
        seating = teams if rng.random() < 0.5 else (teams[1], teams[0])
        d = Duel(random.Random(rng.randrange(1 << 30)), seating,
                 max_turns=max_turns)
        agents = {}
        for side in (0, 1):
            mine = d.teams[side] is engine.MY_TEAM
            name = rng.choice(MINE if mine else THEIRS)
            agents[side] = _agent((ai.MY_POLICIES if mine
                                   else ai.FOE_POLICIES)[name])
        rows = []
        while not d.done:
            side = d.active
            board, scal = d.observe()
            mask = d.legal_mask()
            act = int(agents[side].act_batch([d])[0])
            if mask[act]:                      # skip anything not legal
                rows.append((board, scal, mask, act, side))
            d.step(act)
        for board, scal, mask, act, side in rows:
            B.append((board * 255).astype(np.uint8))
            S.append(scal.astype(np.float16))
            M.append(mask.astype(np.uint8))
            A.append(act)
            # outcome from that side's own point of view
            W.append(d.result if side == 0 else 1.0 - d.result)
    return (np.asarray(B, np.uint8), np.asarray(S, np.float16),
            np.asarray(M, np.uint8), np.asarray(A, np.int16),
            np.asarray(W, np.float16))


def collect(n_games, procs, max_turns, seed=1):
    per = max(1, n_games // procs)
    jobs = [(per, seed + 991 * i, max_turns) for i in range(procs)]
    t0 = time.time()
    with Pool(procs) as pool:
        shards = list(pool.imap_unordered(collect_shard, jobs))
    out = [np.concatenate([s[k] for s in shards]) for k in range(5)]
    print(f'  collected {len(out[3]):,} decisions from {per*procs:,} games '
          f'in {time.time()-t0:.0f}s')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=8000)
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--width', type=int, default=128)
    ap.add_argument('--blocks', type=int, default=6)
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--ckpt', default='checkpoints')
    ap.add_argument('--procs', type=int, default=os.cpu_count() or 4)
    ap.add_argument('--max-turns', type=int, default=60)
    ap.add_argument('--data', default=None,
                    help='reuse a previously collected .npz instead of playing')
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from match_monsters.rl import nets

    os.makedirs(args.ckpt, exist_ok=True)
    cache = args.data or os.path.join(args.ckpt, 'bc_data.npz')
    if args.data and os.path.exists(cache):
        z = np.load(cache)
        B, S, M, A, W = (z['B'], z['S'], z['M'], z['A'], z['W'])
        print(f'  reusing {len(A):,} decisions from {cache}')
    else:
        print(f'playing {args.games:,} games between the hand-tuned policies')
        B, S, M, A, W = collect(args.games, args.procs, args.max_turns)
        np.savez_compressed(cache, B=B, S=S, M=M, A=A, W=W)
        print(f'  saved to {cache}')

    device = nets.pick_device(args.device)
    arch = dict(width=args.width, blocks=args.blocks, hidden=args.hidden)
    net = nets.ActorCritic(**arch).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    n = len(A)
    cut = int(n * 0.97)
    idx_all = np.random.permutation(n)
    tr, va = idx_all[:cut], idx_all[cut:]
    print(f'\ntraining on {len(tr):,} decisions, holding out {len(va):,}')
    print(f'  {sum(p.numel() for p in net.parameters()):,} params on {device}\n')
    print('%6s %11s %11s %11s %9s' % ('epoch', 'train loss', 'val loss',
                                      'val match', 'val value'))

    def batch(ix):
        b = torch.as_tensor(B[ix].astype(np.float32) / 255.0, device=device)
        s = torch.as_tensor(S[ix].astype(np.float32), device=device)
        m = torch.as_tensor(M[ix], device=device)
        a = torch.as_tensor(A[ix].astype(np.int64), device=device)
        w = torch.as_tensor(W[ix].astype(np.float32) * 2 - 1, device=device)
        return b, s, m, a, w

    for ep in range(1, args.epochs + 1):
        net.train()
        np.random.shuffle(tr)
        tot, seen = 0.0, 0
        for k in range(0, len(tr), args.batch):
            ix = tr[k:k + args.batch]
            b, s, m, a, w = batch(ix)
            logits, value = net(b, s, m)
            loss = F.cross_entropy(logits, a) + 0.5 * F.mse_loss(value, w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(ix); seen += len(ix)
        net.eval()
        with torch.no_grad():
            vl, vc, vv, vn = 0.0, 0, 0.0, 0
            for k in range(0, len(va), args.batch):
                ix = va[k:k + args.batch]
                b, s, m, a, w = batch(ix)
                logits, value = net(b, s, m)
                vl += float(F.cross_entropy(logits, a)) * len(ix)
                vc += int((logits.argmax(-1) == a).sum())
                vv += float(F.mse_loss(value, w)) * len(ix)
                vn += len(ix)
        print('%6d %11.4f %11.4f %10.1f%% %9.4f'
              % (ep, tot / seen, vl / vn, 100 * vc / vn, vv / vn))
        torch.save({'net': net.state_dict(), 'opt': opt.state_dict(),
                    'arch': arch, 'iter': 0, 'shared': True,
                    'pretrained': True},
                   os.path.join(args.ckpt, 'selfplay.pt'))

    print(f'\nwrote {os.path.join(args.ckpt, "selfplay.pt")}')
    print('"val match" is how often it picks the same move the hand-tuned')
    print('policy would. Now continue with:  mm-train --resume')


if __name__ == '__main__':
    main()
