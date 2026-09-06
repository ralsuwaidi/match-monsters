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
!pip -q install gymnasium torch numpy
!git clone <your-repo> mm || true
%cd mm
```

Then train:

```python
!python train_coevolve.py --iters 2000 --games 512 --steps 16384 --device cuda
```

Resume from a checkpoint (they are written every iteration to
`checkpoints/coevolve.pt`):

```python
!python train_coevolve.py --iters 4000 --resume --device cuda
```

Mount Drive first if you want checkpoints to survive the runtime being recycled:

```python
from google.colab import drive; drive.mount('/content/drive')
!python train_coevolve.py --iters 2000 --ckpt /content/drive/MyDrive/mm_ckpt --resume
```

## Files needed

`selfplay.py`, `nets.py`, `train_coevolve.py`, `engine.py`, `ai.py`, `grid.py`,
`monsters.py`, `rules.py`, `runner.py`, `progress.py`. Nothing else is imported
at training time.

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
