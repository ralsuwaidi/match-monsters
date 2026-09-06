# Reading the training log

```
  iter    vs-heur    self   turns   match    evo entropy  waste  step/s
   120      34.2%   41.2%    17.1   90.4%   1.47   0.41  0.096    1400
   121      36.0%   41.0%    17.0   90.1%   1.51   0.42  0.096    1401
       evaluation: playing Bonzumi+Sipzap 31%, playing Pelijet+Barbenin 28%
```

The header reprints every 20 rows. A `<-` suffix appears only when something
needs attention.

## The columns

| column | what it is | reference |
|---|---|---|
| `iter` | iteration. One iteration is `--steps` transitions | — |
| `vs-heur` | **win rate against the hand-tuned policy in this iteration's training games.** The headline number once `--vs-heuristic` is on | 50% = matched it |
| `self` | rolling win rate for Bonzumi+Sipzap over the last 4000 games. A measure of the **teams**, not of training | solver says 46.0% |
| `turns` | turns to win, over decisive games. The best single measure of skill | matches-only 26.4, **hand-tuned 16.0** |
| `match` | share of swaps that cleared tiles | random 10.2%, **hand-tuned 96.3%** |
| `evo` | evolutions per game, both sides | **hand-tuned 1.73** |
| `entropy` | policy entropy. Uniform over 60 actions is 4.09 | below 0.3 is a risk |
| `waste` | current wasted-move penalty, annealing to 0 | starts 0.15 |
| `step/s` | throughput | — |

The `evaluation:` line appears every `--eval-every` iterations. It is the same
measurement as `vs-heur` but on fixed games rather than training ones, and it
reports each team separately.

**`turns`, `match` and `evo` count both sides.** The `mm-eval` breakdown reports
one side at a time, so its numbers are roughly half. Don't compare directly.

## Flags

A suffix appears when a metric needs attention:

| flag | meaning |
|---|---|
| `decisive N%` | fewer than 99% of games ended in a kill. Early on this is normal; later it means play has degraded |
| `entropy low` | below 0.3 — the policy has stopped exploring. If the other columns are also flat, that is premature convergence |
| `not matching` | match rate under 20% after iteration 20 — exploration has failed and the agent never found matching |

## What a change means

**`vs-heur` rising** — the only unambiguous progress signal. Everything else can
improve while the agent stays unable to beat a real opponent.

**`turns` falling** — learning *which* match to make. Below 26.4 it beats picking
a random match; at 16.0 it has matched the hand-tuned policy.

**`match` rising** — learning the mechanical game. Moves first and fastest, and
is *not* evidence of strategy: picking random matches already gets 95%.

**`evo` rising** — it found the evolution decision, which has no immediate
payoff: it costs a move and changes nobody's health. Skipping evolution costs
the hand-tuned policy 11.7 points.

**`entropy` falling** — committing to a strategy. Healthy while other columns
improve; a warning when they have stopped.

**`self` drifting** — with one network playing both teams this measures the
**rosters**. Discount it while the network is weak: its known weaknesses are
team-specific, so a poorly-trained agent makes whichever team it handles worse
look worse than it is.

## Warning signs

| symptom | meaning |
|---|---|
| `match` stuck near 10% | exploration failed; the agent never found matching |
| `match` collapses as `waste` reaches 0 | matching was enforced, never learned |
| `decisive` stuck near 0% | no terminal signal; nothing to learn from winning |
| `entropy` under 0.3 with `turns` flat | premature convergence — roll back to a snapshot and raise `--entropy` |
| `vs-heur` flat while `turns` improves | it is getting better at beating itself, not at playing |
