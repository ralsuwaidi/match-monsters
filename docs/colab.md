# Running the co-evolution on Google Colab

The code is device-agnostic — `nets.pick_device()` picks `cuda`, then `mps`,
then `cpu`. The same command runs on your laptop and on a Colab GPU.

## Network size vs speed (measured on an M-series laptop, mps)

| width / blocks / hidden | params per side | steps/sec | 20M steps |
|---|---|---|---|
| 64 / 3 / 256   | 965,629   | 1,651 | 3.4 h |
| 96 / 4 / 384   | 2,250,397 | 803   | 6.9 h |
| 128 / 6 / 512  | 4,519,357 | 489   | 11.4 h |

**This is where a Colab GPU changes the picture.** With the small net the Python
engine dominates and CUDA barely helps. At 4.5M parameters the *network*
dominates, so a T4/A100 gives a large speedup — use `just train-big` there and
the laptop-sized net locally.

## What actually limits speed

The bottleneck is the **pure-Python game engine**, not the network. Measured on
this machine:

| stage | rate |
|---|---|
| engine + observation encoding, no net | ~46,000 steps/sec |
| rollout with batched inference (mps, 512 games) | ~25,000 steps/sec |
| full training loop incl. PPO update | ~3,200 steps/sec |

So a Colab **GPU speeds the PPO update, not the rollouts**. Expect maybe
1.5–2× end to end from a T4/A100, not 10×. The larger win on Colab is **more
CPU cores** for parallel rollouts, or porting `grid.py` to Rust.

Raise `--games` on a GPU: bigger inference batches amortise the transfer cost.
512 games was ~25k steps/sec on mps versus ~7.5k at 32 games.

## Colab setup

```python
# one cell
!git clone <your-repo> mm && cd mm && pip -q install -e ".[rl]"
%cd mm
```

Then train:

```python
!mm-train --iters 2000 --games 512 --steps 16384 --device cuda \
          --width 128 --blocks 6 --hidden 512
```

The exploration settings (`--waste-penalty 0.15 --match-curriculum 0.6`) are
the defaults, so they do not need passing. Check the second line the run
prints: if it says `waste_penalty 0.05`, you are on an old checkout and should
pull -- 0.05 is a setting that was measured and does not work.

Resume from a checkpoint (written every iteration to `checkpoints/selfplay.pt`):

```python
!mm-train --iters 4000 --resume --device cuda
```

Mount Drive first if you want checkpoints to survive the runtime recycling:

```python
from google.colab import drive; drive.mount('/content/drive')
!mm-train --iters 2000 --ckpt /content/drive/MyDrive/mm_ckpt --resume
```

## Files needed

The whole `src/match_monsters` package -- it is installed by `pip install -e .`
so the imports resolve. Nothing outside it is needed at training time.

## Getting the run reviewed

Every run writes `runs/<run-id>/train.log` and `runs/<run-id>/progress.json`.
For a compact summary to paste back for review:

```python
!mm-report-run
```

It prints the settings, the learning curve, and the reference points that say
what the numbers mean. On Colab, copy `runs/` to Drive if you want it to
survive the runtime being recycled.

## Reading the output

```
iter  120  side-0 win  46.5%  games  128  steps 1,491,000  3238/s  entropy 3.88/3.83
```

- **side-0 win** — rolling win rate of Bonzumi + Sipzap against Pelijet +
  Barbenin, both learning. This is the number that should converge.
- **entropy** — per side, starts near ln(60) = 4.09 for a uniform policy and
  falls as each agent commits to a strategy. If it collapses below ~1.0 early,
  raise `--entropy`.
- A timeout is graded by HP difference rather than scored 0.5, and there is a
  dense `--shaping` reward on HP swing. Both exist because two random agents
  otherwise draw every game and never generate a learning signal.
