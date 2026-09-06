# Match Monsters — simulator, solver and self-play

Answers one question with evidence rather than intuition: **is Bonzumi + Sipzap
good enough against Pelijet + Barbenin?**

**Short answer: 46.0%** — a coin flip, very slightly unfavourable, measured over
2.35 million simulated duels with both sides playing their best. Full reasoning
and every supporting number is in **[docs/findings.md](docs/findings.md)**.

```bash
just            # list every command
just setup      # install
just test       # 16 tests covering the rules that must hold
just play       # play a game yourself against the strongest enemy policy
just report     # the whole analysis
```

## What is here

Three layers, each usable on its own.

### 1. A validated rules engine

`rules.py` `grid.py` `monsters.py` `engine.py` — the game itself. Every rule is
marked `[C]` (confirmed against the official Plato rules, the official monster
descriptions, or checked in game) or `[A]` (assumption, and measured in the
sensitivity report).

It is checked against real boards from screenshots — `just validate` — and
matches on every count: no board carries a pre-existing match, and the number of
available matches sits on the simulated median.

### 2. A policy solver

`ai.py` `runner.py` `solver.py` `app.py` — hand-tuned policies scored in
damage-equivalent units, and a solver that finds the best one by **racing**:
every policy starts with a small budget, and any policy whose best possible
floor falls below the leader's worst possible floor is eliminated on confidence
intervals. That reached the same answer with 113k duels where brute force needed
432k.

```bash
just solve      # detached; survives closing the terminal
just ui         # live dashboard at localhost:8501
```

### 3. Neural self-play

`selfplay.py` `nets.py` `train_selfplay.py` — one network playing **both** teams
and improving against itself. The observation is side-relative and describes
each monster by its *stats* — cost, damage, whether it charges, what its ability
does — never by name, so the same weights adapt to whichever side they are
playing.

Because the identical brain plays both teams, any deviation from a 50% win rate
is a property of the **rosters**, not of one side being trained harder.

```bash
just baseline   # what random / match-only / heuristic play look like
just train      # start training, detached
just eval       # is the network any good yet?
```

**Status: it does not work yet.** See [Honest status](#honest-status).

## Reading a training number

`just baseline` prints the reference points, and they are the only way to know
what a training number means:

| policy | damage/game | turns to win | match rate | decisive |
|---|---|---|---|---|
| uniform over legal moves | 24.7 | 60.0 (timeout) | 10.2% | 0% |
| uniform over **matches only** | 141.5 | 26.4 | 95.0% | 100% |
| hand-tuned heuristic | 142.7 | **16.0** | 96.3% | 100% |

Almost all the value is in one binary skill — make a match instead of a random
swap. Everything the solver measures (denial, honey timing, berry setups) is the
26.4 → 16.0 turn improvement layered on top.

Win rate is a poor progress metric in self-play: it sits near 50% whether both
agents are brilliant or both are useless. Watch **turns to win** and **match
rate**, and the absolute yardstick — the network played against the fixed
hand-tuned policy.

## Honest status

The neural agent is **stuck at the random baseline** (10% match rate) after
every configuration tried. The cause is exploration, not capacity: a match is
about 5 of 49 legal actions and two or three are needed before any reward
appears, so random search never finds the chain.

Two curricula were tried and both failed:

- **Masking to matching moves only** — the match rate tracked the mask exactly
  and collapsed the moment it lifted. Masked logits are set to `-inf`, never
  sampled, never updated, so they spring back to their initialisation.
- **An annealed penalty for wasted moves** — gives the bad actions gradient, but
  peaked at 14% and fell back.

Both experiments ran ~500k steps. Typical PPO needs 10⁷–10⁸. **So "wrong
hyperparameters" and "not enough compute" both fit the data**, and the runs so
far cannot distinguish them. The next honest experiment is a multi-hour run —
which is what [docs/colab.md](docs/colab.md) is for.

## Layout

```
src/match_monsters/
├── rules.py              every rule constant, [C]onfirmed or [A]ssumed
├── game/                 the rules engine
│   ├── grid.py             board: tiles, swaps, gravity, refill, cascades
│   ├── monsters.py         monster stats and abilities
│   └── engine.py           sides, mana, firing, turns, a duel
├── agents/
│   └── ai.py               valuation, policies, move choice
├── solver/
│   ├── runner.py           multiprocessing and confidence intervals
│   ├── race.py             the racing policy solver
│   ├── evolve.py           co-evolution of the policy weights
│   └── progress.py         run state, shared with the dashboard
├── rl/
│   ├── selfplay.py         two-sided game state for neural play
│   ├── nets.py             actor-critic, configurable width and depth
│   ├── train.py            PPO self-play, one net playing both teams
│   ├── evaluate.py         net vs net, net vs heuristic
│   ├── baselines.py        the reference points above
│   └── report.py           pasteable summary of a run
├── analysis/
│   ├── reports.py          board stats, matrix, headline, sensitivity
│   ├── validate.py         engine checked against real screenshots
│   ├── berry_sweep.py      re-solve at every plausible berry rate
│   ├── berry_calibrate.py  board counts -> spawn-rate estimate
│   ├── rollout.py          which line is better from a live position
│   ├── trace.py            print a full game turn by turn
│   └── head_to_head.py     detailed matchup stats for the dashboard
├── ui/app.py             the Streamlit dashboard
├── play.py               play a game yourself in the terminal
└── sim.py                report entry point

tests/                    rules that must hold -- `just test`
docs/                     findings.md, colab.md
```

Every command is a `just` recipe and also a console script (`mm-play`,
`mm-solve`, `mm-train`, ...) installed by `uv sync`.
