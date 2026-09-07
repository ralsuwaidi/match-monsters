# Match Monsters -- run `just` to list everything.
#
#   src/match_monsters/game      the rules engine
#   src/match_monsters/agents    hand-tuned policies
#   src/match_monsters/solver    policy search + dashboard
#   src/match_monsters/rl        neural self-play
#   src/match_monsters/analysis  reports and studies

uv := "uv run"
rl := "uv run --extra rl"
app := "src/match_monsters/ui/app.py"

# show every recipe
default:
    @just --list --unsorted

# ---------------------------------------------------------------- setup --

# install the base dependencies
setup:
    uv sync

# install everything, including torch for the neural training
setup-rl:
    uv sync --extra rl --extra dev

# run the test suite
test:
    {{uv}} --extra dev pytest tests/ -q

# what device the neural code will use here
device:
    @{{rl}} python -c "from match_monsters.rl import nets; import torch; print('torch', torch.__version__, '->', nets.pick_device())"

# ------------------------------------------------------------- play it --

# play a game yourself against the strongest enemy policy
play:
    {{uv}} mm-play play

# play a specific board, e.g.  just play-seed 424242
play-seed seed:
    {{uv}} mm-play play {{seed}}

# play against a different enemy policy
play-vs seed policy:
    {{uv}} mm-play play {{seed}} {{policy}}

# print a full simulated game, turn by turn
trace seed="20260905":
    {{uv}} mm-trace {{seed}}

# ------------------------------------------------------- the solver ----

# live dashboard -- start/stop runs, watch the matrix fill in
ui:
    {{uv}} streamlit run {{app}}

# find the best policy. detached: closing the terminal will not stop it
solve start="800" rounds="7":
    nohup {{uv}} mm-solve --start {{start}} --rounds {{rounds}} \
        --berry 0.10 --carryover off --base-hp 80 > /tmp/mm_solver.log 2>&1 &
    @sleep 2 && echo "started; watch with 'just solve-log' or 'just ui'"

solve-log:
    @tail -f /tmp/mm_solver.log

solve-stop:
    -@pkill -f "solver.race\|mm-solve" && echo stopped || echo "nothing running"

# co-evolve the heuristic WEIGHTS (much cheaper than neural training)
evolve gens="40":
    nohup {{uv}} mm-evolve --gens {{gens}} --berry 0.10 --carryover off \
        > /tmp/mm_evolve.log 2>&1 &
    @sleep 2 && echo "started; watch with 'just evolve-log'"

evolve-log:
    @tail -f /tmp/mm_evolve.log

# ------------------------------------------------------- the reports ----

# everything: board stats, policy matrix, headline, sensitivity, AI strength
report:
    {{uv}} mm-report

# one section only: board | matrix | head | sens | ai
report-only section:
    {{uv}} mm-report {{section}}

# which line is better from the position saved in game_state.json
rollout n="2500":
    {{uv}} mm-rollout {{n}}

# re-solve the matchup across every plausible berry spawn rate
berry-sweep:
    {{uv}} mm-berry-sweep

# turn observed on-board berry counts into a spawn-rate estimate
berry-calibrate:
    {{uv}} mm-berry-calibrate

# check the engine against the real boards from the screenshots
validate:
    {{uv}} mm-validate

# ------------------------------------------ neural self-play ----

# STEP 1: learn to imitate the hand-tuned policies before any RL. Starting
# from random weights and trying to discover competent play by exploration is
# what failed repeatedly; this skips it.
pretrain games="40000":
    @mkdir -p logs
    nohup {{rl}} python -u -m match_monsters.rl.pretrain --games {{games}} \
        --epochs 12 --width 64 --blocks 3 --hidden 256 > logs/pretrain.log 2>&1 &
    @sleep 2 && echo "watch it with 'just pretrain-log'"

# watch the cloning run
pretrain-log:
    @tail -f logs/pretrain.log

# reuse the collected demonstrations, just retrain (skips the 100s of play)
pretrain-again epochs="12" width="64" blocks="3" hidden="256":
    {{rl}} python -u -m match_monsters.rl.pretrain --data checkpoints/bc_data.npz \
        --epochs {{epochs}} --width {{width}} --blocks {{blocks}} --hidden {{hidden}}

# STEP 2: ONE network playing both teams and improving against itself (detached).
# It sees each monster's stats, not its name, so the same weights adapt to
# whichever side it is playing.
train iters="1500":
    nohup {{rl}} mm-train --iters {{iters}} > /tmp/mm_nn.log 2>&1 &
    @sleep 2 && echo "training started; watch with 'just train-log'"

# foreground, so you can watch it and Ctrl-C
train-fg iters="50":
    {{rl}} mm-train --iters {{iters}}

# continue from the last checkpoint (falls back to a numbered snapshot if the
# latest is unreadable)
train-resume iters="4000":
    nohup {{rl}} mm-train --iters {{iters}} --resume > /tmp/mm_nn.log 2>&1 &
    @sleep 2 && echo "resumed; watch with 'just train-log'"

# the big network -- only worth it on a CUDA GPU, see docs/colab.md
train-big iters="4000":
    nohup {{rl}} mm-train --iters {{iters}} \
        --width 160 --blocks 8 --hidden 768 --games 512 --steps 16384 \
        > /tmp/mm_nn.log 2>&1 &
    @sleep 2 && echo "training started; watch with 'just train-log'"

# a specific size, e.g.  just train-size 2000 128 6 512
train-size iters width blocks hidden:
    nohup {{rl}} mm-train --iters {{iters}} --width {{width}} \
        --blocks {{blocks}} --hidden {{hidden}} > /tmp/mm_nn.log 2>&1 &
    @sleep 2 && echo "training started; watch with 'just train-log'"

# how many parameters each size gives you
train-sizes:
    @{{rl}} python -c "from match_monsters.rl import nets; [print('width %3d blocks %d hidden %3d -> %s params'%(w,b,h,format(sum(p.numel() for p in nets.ActorCritic(w,b,h).parameters()),','))) for w,b,h in [(64,3,256),(96,4,384),(128,6,512),(160,8,768)]]"

train-log:
    @tail -f /tmp/mm_nn.log

train-stop:
    -@pkill -f "rl.train\|mm-train" && echo stopped || echo "nothing running"

# PASTE THIS BACK for review -- settings, learning curve, and what it means
report-run run="":
    @{{rl}} mm-report-run {{run}}

# what random / match-only / heuristic play look like: the reference points
baseline:
    @{{rl}} mm-baseline

# is the network any good yet, against the hand-tuned policies?
eval games="400":
    {{rl}} mm-eval --games {{games}}

# the same network used as an EVALUATOR with lookahead, rather than as a policy
eval-search games="200" samples="2":
    {{rl}} mm-eval --games {{games}} --search {{samples}}

# ------------------------------------------------------------ tidy ----

# stop everything running in the background
stop-all:
    -@pkill -f "mm-train\|mm-solve\|mm-evolve\|rl.train\|solver.race\|streamlit"; echo "all stopped"

# delete run artefacts, game logs and checkpoints
clean:
    rm -rf runs games checkpoints game_state.json
    @echo "cleaned"
