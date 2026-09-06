# Match Monsters simulator -- run `just` to list everything.
#
# Two separate things live here:
#   1. a validated rules engine + hand-tuned policy solver  (no extra deps)
#   2. neural co-evolution self-play                        (needs --extra rl)

uv := "uv run"
rl := "uv run --extra rl"

# show every recipe
default:
    @just --list --unsorted

# ---------------------------------------------------------------- setup --

# install the base dependencies (streamlit, pandas)
setup:
    uv sync

# install everything including torch + gymnasium for the neural training
setup-rl:
    uv sync --extra rl

# what device the neural code will use here
device:
    @{{rl}} python -c "import nets, torch; print('torch', torch.__version__, '->', nets.pick_device())"

# ------------------------------------------------------------- play it --

# play a game yourself against the strongest enemy policy (random board)
play:
    {{uv}} python play.py play

# play a specific board, e.g.  just play-seed 424242
play-seed seed:
    {{uv}} python play.py play {{seed}}

# play against a different enemy policy, e.g.  just play-vs 424242 pel_hdeny
play-vs seed policy:
    {{uv}} python play.py play {{seed}} {{policy}}

# print a full simulated game, turn by turn, with the board at every turn
trace seed="20260905":
    {{uv}} python trace.py {{seed}}

# ------------------------------------------------------- the solver ----

# live dashboard: start/stop runs, watch the matrix fill in  (localhost:8501)
ui:
    {{uv}} streamlit run app.py

# solve for the best policy. detached -- closing the terminal will not stop it
solve start="800" rounds="7":
    nohup {{uv}} python solver.py --start {{start}} --rounds {{rounds}} \
        --berry 0.10 --carryover off --base-hp 80 > /tmp/mm_solver.log 2>&1 &
    @sleep 2 && echo "started; watch with 'just solve-log' or 'just ui'"

solve-log:
    @tail -f /tmp/mm_solver.log

# stop any running solver
solve-stop:
    -@pkill -f "solver.py" && echo stopped || echo "nothing running"

# ------------------------------------------------------- the reports ----

# everything: board stats, policy matrix, headline, sensitivity, AI strength
report:
    {{uv}} python sim.py

# board statistics only -- match availability, deadlock rate
report-board:
    {{uv}} python sim.py board

# how the win rate moves if a single assumption is wrong
report-sens:
    {{uv}} python sim.py sens

# which line is better from the position saved in game_state.json
rollout n="2500":
    {{uv}} python rollout.py {{n}}

# re-solve the matchup across every plausible berry spawn rate
berry-sweep:
    {{uv}} python berry_sweep.py

# turn observed on-board berry counts into a spawn-rate estimate
berry-calibrate:
    {{uv}} python berry_calibrate.py

# check the engine against the real boards from the screenshots
validate:
    {{uv}} python validate_real.py

# ------------------------------------------ neural co-evolution ----

# ONE network playing both teams and improving against itself (detached).
# It sees each monster's stats, not its name, so the same weights adapt to
# whichever side it is playing. 2.25M params, ~870 steps/sec on a laptop.
train iters="1500":
    nohup {{rl}} python train_selfplay.py --iters {{iters}} \
        --waste-penalty 0.15 --match-curriculum 0.6 > /tmp/mm_nn.log 2>&1 &
    @sleep 2 && echo "training started; watch with 'just train-log'"

# foreground so you can watch it and Ctrl-C
train-fg iters="50":
    {{rl}} python train_selfplay.py --iters {{iters}}

# continue from checkpoints/selfplay.pt
train-resume iters="4000":
    nohup {{rl}} python train_selfplay.py --iters {{iters}} --resume > /tmp/mm_nn.log 2>&1 &
    @sleep 2 && echo "resumed; watch with 'just train-log'"

# the big net -- only worth it on a CUDA GPU, see COLAB.md
train-big iters="4000":
    nohup {{rl}} python train_selfplay.py --iters {{iters}} \
        --width 160 --blocks 8 --hidden 768 --games 512 --steps 16384 \
        > /tmp/mm_nn.log 2>&1 &
    @sleep 2 && echo "training started; watch with 'just train-log'"

# specific size, e.g.  just train-size 2000 128 6 512
train-size iters width blocks hidden:
    nohup {{rl}} python train_selfplay.py --iters {{iters}} --width {{width}} \
        --blocks {{blocks}} --hidden {{hidden}} > /tmp/mm_nn.log 2>&1 &
    @sleep 2 && echo "training started; watch with 'just train-log'"

# the older design: two separate networks, one per team
train-two iters="1600":
    nohup {{rl}} python train_coevolve.py --iters {{iters}} \
        --width 64 --blocks 3 --hidden 256 > /tmp/mm_nn.log 2>&1 &
    @sleep 2 && echo "training started; watch with 'just train-log'"

# how many parameters each network size gives you
train-sizes:
    @{{rl}} python -c "import nets; [print('width %3d blocks %d hidden %3d -> %s params'%(w,b,h,format(sum(p.numel() for p in nets.ActorCritic(w,b,h).parameters()),','))) for w,b,h in [(64,3,256),(128,6,512),(192,8,768),(256,10,1024)]]"

train-log:
    @tail -f /tmp/mm_nn.log

train-stop:
    -@pkill -f "train_selfplay.py"; pkill -f "train_coevolve.py"; echo stopped

# what random / match-only / heuristic play look like -- the skill ceiling
baseline:
    @{{rl}} python baseline.py

# are the networks any good yet? both nets head to head, and vs the heuristics
eval games="400":
    {{rl}} python eval_nn.py --games {{games}}

# same, but sampling from the policy rather than always taking the argmax
eval-sample games="400":
    {{rl}} python eval_nn.py --games {{games}} --sample

# co-evolve the HEURISTIC weights instead of neural nets (much cheaper)
coevolve gens="40":
    nohup {{uv}} python coevolve.py --gens {{gens}} --berry 0.10 --carryover off \
        > /tmp/mm_coev.log 2>&1 &
    @sleep 2 && echo "started; watch with 'just coevolve-log'"

coevolve-log:
    @tail -f /tmp/mm_coev.log

# ------------------------------------------------------------ tidy ----

# stop everything that is running in the background
stop-all:
    -@pkill -f solver.py; pkill -f train_coevolve.py; pkill -f coevolve.py; \
      pkill -f berry_sweep.py; echo "all stopped"

# delete solver runs, game logs and checkpoints
clean:
    rm -rf runs games checkpoints game_state.json
    @echo "cleaned"
