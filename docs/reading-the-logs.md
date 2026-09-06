# Reading the training log

```
iter  315  BS  44.3% PB  55.7%  n=409  roll  41.3%  turns  17.9  match 87.3%
           dmg 143.1  fires 13.1  decisive 100%  evo 1.44  wp 0.111  1401/s  ent 0.43
```

Every field, what it means, and what a change in it is telling you.

## The fields

| field | what it is | reference |
|---|---|---|
| `iter` | iteration number. One iteration is `--steps` transitions | — |
| `BS` / `PB` | win rate **this iteration** for Bonzumi+Sipzap and Pelijet+Barbenin. Noisy; read `roll` instead | 50% = balanced |
| `n` | games that finished this iteration. **Small `n` makes `BS` meaningless** | 300-500 is solid |
| `roll` | rolling win rate over the last 4000 games, from Bonzumi+Sipzap's side. **This is the trustworthy one** | solver says 46.0% |
| `turns` | turns to win, averaged over decisive games. **The best single measure of skill** | random-over-matches 26.4, **heuristic 16.0** |
| `match` | share of swaps that cleared tiles. The rest were repositions or wasted | random 10.2%, **heuristic 96.3%** |
| `dmg` | damage per game, **both sides combined** | ceiling ~142.7 |
| `fires` | monster activations per game, both sides | heuristic ~11 |
| `decisive` | games ending in a kill rather than the 60-turn timeout | must reach 100% |
| `evo` | evolutions per game, **both sides combined** | **heuristic 1.73** |
| `wp` | current waste penalty, annealing to 0 | starts 0.15 |
| `N/s` | training throughput | — |
| `ent` | policy entropy. Uniform over 60 actions is ln(60) = 4.09 | collapse below ~0.3 is a risk |
| `vs heuristic` | evaluated every `--eval-every`: the network against the hand-tuned policy, playing each team | 50% = matched it |

**`dmg`, `fires` and `evo` count both sides.** The `mm-eval` breakdown reports
one side at a time, so its numbers are roughly half these. Do not compare them
directly.

## What a change means

**`match` rising** — the agent is learning the mechanical game. This moves first
and fastest. It is *not* evidence of strategy; random-over-matches gets 95%.

**`turns` falling** — the agent is learning *which* match to make. This is the
real skill signal. Below 26.4 it beats random-over-matches; at 16.0 it has
matched the hand-tuned policy.

**`decisive` rising to 100%** — games are being won rather than timing out. Until
this happens the terminal reward is near zero and almost nothing is being
learned from winning.

**`evo` rising** — it has found the evolution decision, which has *no immediate
payoff*: it costs a move and changes nobody's HP. Skipping evolution costs the
hand-tuned policy 11.7 points of win rate, so an agent stuck near 0 is leaving a
lot behind.

**`dmg` at ~142 but `turns` still high** — it generates plenty of mana and
converts it badly. Look at `mm-eval` for damage-per-mana and the per-monster
split.

**`ent` falling** — the policy is committing. Healthy while other metrics
improve. If `ent` keeps dropping while `match` and `turns` have stalled, that is
premature convergence: roll back to a numbered snapshot and raise `--entropy`.

**`roll` drifting** — with one network playing both teams, this is a measurement
of the **rosters**, not of training. Expect it to be noisy early and to settle as
`turns` plateaus. Discount it entirely while `wp` and shaping are still large,
because both distort what the agent optimises.

## Warning signs

| symptom | meaning |
|---|---|
| `match` stuck near 10% | exploration failed; the agent never found matching |
| `match` collapses as `wp` reaches 0 | matching was only enforced, never learned |
| `decisive` stuck at 0% | no terminal signal; nothing to learn from |
| `ent` below 0.3 with metrics flat | premature convergence |
| `n` small and `BS` swinging wildly | sample-size noise, not a real change |
