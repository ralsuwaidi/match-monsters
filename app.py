"""Live view of the policy solver.

    uv run streamlit run app.py

The solver runs fully detached. Closing this page, or stopping Streamlit
entirely, does not touch it -- the UI only ever reads runs/<id>/progress.json.
Reattach any time by reopening and picking the run.
"""
import os
import signal
import subprocess
import sys
import time

import pandas as pd
import streamlit as st

import ai
import analysis
import progress

HERE = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="Match Monsters solver", layout="wide")


def gate():
    pw = os.environ.get("MM_PASSWORD")
    if not pw or st.session_state.get("ok"):
        return True
    st.title("Match Monsters solver")
    entered = st.text_input("Password", type="password")
    if entered:
        if entered == pw:
            st.session_state["ok"] = True
            st.rerun()
        st.error("Wrong password")
    st.caption("Set MM_PASSWORD to require a password; unset it to disable this.")
    return False


if not gate():
    st.stop()


def launch(start, rounds, keep, berry, carry, hp):
    run_id = progress.new_run_id()
    os.makedirs(progress.run_dir(run_id), exist_ok=True)
    log = open(progress.log_path(run_id), "wb")
    cmd = [sys.executable, os.path.join(HERE, "solver.py"),
           "--run-id", run_id, "--start", str(start), "--rounds", str(rounds),
           "--keep", str(keep), "--berry", str(berry),
           "--carryover", "on" if carry else "off", "--base-hp", str(hp)]
    # start_new_session detaches it from Streamlit's process group, so closing
    # the UI -- or the whole Streamlit server -- leaves the solver running
    subprocess.Popen(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True)
    return run_id


def fmt_secs(s):
    if not s or s != s:
        return "-"
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    return f"{s // 3600}h {(s % 3600) // 60}m"


def cellfmt(v):
    return "·" if pd.isna(v) else f"{v:.1f}"


def shade(v):
    """Green above 50%, red below. Avoids a matplotlib dependency."""
    if pd.isna(v):
        return "background-color: rgba(128,128,128,0.10)"
    t = max(0.0, min(1.0, (v - 30.0) / 40.0))
    if t >= 0.5:
        return f"background-color: rgba(35,160,90,{0.10 + 0.45 * (t - 0.5) * 2:.2f})"
    return f"background-color: rgba(200,60,60,{0.10 + 0.45 * (0.5 - t) * 2:.2f})"


# ------------------------------------------------------------- sidebar ----
st.sidebar.header("Runs")
runs = progress.list_runs()
sel = st.sidebar.selectbox("View run", runs, index=0) if runs else None
if not runs:
    st.sidebar.info("No runs yet — start one below.")

with st.sidebar.form("new"):
    st.subheader("New run")
    start = st.number_input("Duels per cell, round 1", 100, 20000, 400, step=100)
    rounds = st.number_input("Doubling rounds", 1, 10, 6)
    keep = st.number_input("Minimum policies kept", 1, 9, 2)
    berry = st.number_input("Berry spawn rate", 0.0, 0.5, 0.10, step=0.01, format="%.3f")
    carry = st.checkbox("Mana carries over", value=False)
    hp = st.number_input("Base HP", 20, 200, 80, step=5)
    if st.form_submit_button("Start detached run", type="primary"):
        rid = launch(start, rounds, keep, berry, carry, hp)
        st.sidebar.success(f"Started {rid}")
        time.sleep(1.0)
        st.rerun()

auto = st.sidebar.checkbox("Auto-refresh", value=True)
every = st.sidebar.slider("Refresh seconds", 1, 30, 4)
st.sidebar.caption(
    "The solver is detached. Closing this page or killing Streamlit leaves it "
    "running — reopen and pick the run to reattach."
)

if not sel:
    st.title("Match Monsters solver")
    st.write("Start a run from the sidebar.")
    st.stop()

state = progress.read(sel)
if not state:
    st.error(f"Could not read run {sel}")
    st.stop()

kind = state.get("kind", "solver")
running = state["status"] == "running" and progress.alive(state.get("pid"))


def stop_button(state):
    if st.button("Stop this run"):
        try:
            os.kill(state["pid"], signal.SIGTERM)
            st.success("Stop signal sent")
            time.sleep(1.5)
            st.rerun()
        except OSError as e:
            st.error(f"Could not stop: {e}")


def render_neural(state):
    st.title("Neural co-evolution — both sides learning from scratch")
    c = state["config"]
    st.caption(
        f"run `{state['run_id']}`  ·  device {state.get('device','?')}  ·  "
        f"{c['params']:,} params per side  ·  {c['games']} parallel games  ·  "
        f"{c['steps']:,} steps/side/iteration  ·  lr {c['lr']}  ·  "
        f"entropy bonus {c['entropy']}  ·  board {c['board']}  ·  "
        f"berries {100*c['berry_rate']:.0f}%"
    )
    hist = pd.DataFrame(state.get("history", []))
    m = st.columns(5)
    m[0].metric("Status", state["status"])
    m[1].metric("Iteration", f"{state.get('iter',0)} / {state.get('iters_planned','?')}")
    if not hist.empty:
        m[2].metric("Side-0 win rate", f"{hist['wr_rolling'].iloc[-1]:.1f}%")
        m[3].metric("Steps", f"{int(hist['steps'].iloc[-1]):,}")
        m[4].metric("Steps/sec", f"{hist['sps'].iloc[-1]:,.0f}")
    if running:
        stop_button(state)
    if hist.empty:
        st.info("no iterations recorded yet")
        return

    st.subheader("Convergence — Bonzumi + Sipzap win rate against Pelijet + Barbenin")
    st.line_chart(hist.set_index("iter")[["wr_rolling"]].rename(
        columns={"wr_rolling": "side-0 win %"}), height=280)
    st.caption(
        "Both networks are training at once, so this is not one agent getting "
        "better against a fixed opponent — it is the balance between two agents "
        "that are both improving. If it flattens, that level is a property of "
        "the two ROSTERS rather than of either agent's tuning. The hand-tuned "
        "heuristic equilibrium, for comparison, is 46.0%."
    )

    ev = hist.dropna(subset=["A_vs_heuristic"]) if "A_vs_heuristic" in hist else pd.DataFrame()
    if not ev.empty:
        st.subheader("Absolute yardstick — each net against the hand-tuned policy")
        st.line_chart(ev.set_index("iter")[["A_vs_heuristic", "B_vs_heuristic"]].rename(
            columns={"A_vs_heuristic": "net A vs heuristic pel_deny",
                     "B_vs_heuristic": "net B vs heuristic bon_hdeny"}), height=280)
        st.caption(
            "**This is the chart that matters.** The self-play win rate above is "
            "*relative* — it sits near 50% whether both agents are brilliant or "
            "both are useless. These lines are each network against a fixed "
            "opponent that never changes, so they show real progress. 50% here "
            "means the network has matched the hand-tuned policy; above 50% "
            "means it found something better."
        )
        last = ev.iloc[-1]
        k = st.columns(2)
        k[0].metric("net A vs heuristic", f"{last['A_vs_heuristic']:.0f}%")
        k[1].metric("net B vs heuristic", f"{last['B_vs_heuristic']:.0f}%")

    st.subheader("Policy entropy — how committed each side has become")
    ent = hist.set_index("iter")[["entropy_A", "entropy_B"]].replace(0.0, float("nan"))
    ent = ent.rename(columns={"entropy_A": "Bonzumi + Sipzap",
                              "entropy_B": "Pelijet + Barbenin"})
    st.line_chart(ent, height=260)
    st.caption(
        "Starts near ln(60) = 4.09, which is a uniform random policy over the "
        "60 actions. Falling entropy means the agent is committing to a "
        "strategy. While these sit near 4.0 the agents are still essentially "
        "playing at random and the win rate above means very little. If either "
        "collapses below ~1.0 early, raise the entropy bonus. Only the side "
        "learning that iteration reports a value — league play alternates them."
    )

    st.subheader("Value loss — is the critic learning to predict outcomes?")
    st.line_chart(hist.set_index("iter")[["vloss_A", "vloss_B"]].rename(
        columns={"vloss_A": "Bonzumi + Sipzap", "vloss_B": "Pelijet + Barbenin"}),
        height=220)

    with st.expander("Evaluate the checkpoint"):
        st.markdown(
            "The win rate above is the two networks against **each other**. "
            "To find out whether either is actually any good, play them against "
            "the hand-tuned policies:\n\n```\njust eval\n```\n\n"
            "That reports net A vs net B, each net against its heuristic "
            "counterpart, and heuristic vs heuristic as the baseline."
        )
    with st.expander("Raw history"):
        st.dataframe(hist, use_container_width=True, hide_index=True)


def render_coevolve(state):
    st.title("Weight co-evolution — both sides searching for a best response")
    c = state["config"]
    st.caption(f"run `{state['run_id']}`  ·  {c['duels_per_eval']} duels per "
               f"evaluation  ·  {c['lam']} candidates/side/generation  ·  "
               f"pool {c['pool']}  ·  berries {100*c['berry_rate']:.0f}%")
    hist = pd.DataFrame(state.get("history", []))
    m = st.columns(4)
    m[0].metric("Status", state["status"])
    m[1].metric("Generation", f"{state.get('gen',0)} / {state.get('gens_planned','?')}")
    m[2].metric("Duels", f"{state.get('duels',0):,}")
    if not hist.empty:
        m[3].metric("Side-0 win", f"{hist['wr'].iloc[-1]:.1f}%")
    if running:
        stop_button(state)
    if hist.empty:
        st.info("no generations yet")
        return
    st.subheader("Champion vs champion, each generation")
    st.line_chart(hist.set_index("gen")[["wr"]].rename(
        columns={"wr": "side-0 win %"}), height=280)
    st.caption("Each side searches for the best answer to the other's pool of "
               "past champions. When neither can improve, the step size shrinks "
               "and the run converges.")
    st.subheader("Evolved weights")
    for who, label in (("me", "Bonzumi + Sipzap"), ("foo", "x")):
        pass
    cols = st.columns(2)
    for col, key, label in ((cols[0], "me", "Bonzumi + Sipzap"),
                            (cols[1], "foe", "Pelijet + Barbenin")):
        g = state.get(key, {})
        col.markdown(f"**{label}**")
        col.dataframe(pd.DataFrame(
            [{"weight": k, "value": round(v, 3)} for k, v in g.items()
             if k != "order"]
            + [{"weight": "evolve order", "value": " then ".join(g.get("order", []))}]
        ), use_container_width=True, hide_index=True)
    with st.expander("Raw history"):
        st.dataframe(hist, use_container_width=True, hide_index=True)


if kind == "neural":
    render_neural(state)
    if auto and running:
        time.sleep(every)
        st.rerun()
    st.stop()
if kind == "coevolve":
    render_coevolve(state)
    if auto and running:
        time.sleep(every)
        st.rerun()
    st.stop()
if state["status"] == "running" and not running:
    st.warning("Marked running, but the process is gone — it was killed.")

# ---------------------------------------------------------------- head ----
st.title("Match Monsters — policy solver")
cfg = state["config"]
st.caption(
    f"run `{sel}`  ·  board {cfg['board']}  ·  berries {100 * cfg['berry_rate']:.0f}%  ·  "
    f"mana carryover {'on' if cfg['mana_carryover'] else 'off'}  ·  "
    f"HP {cfg['base_hp']} (+{cfg['second_player_hp_bonus']} going second)  ·  "
    f"{cfg['moves_per_turn']} moves/turn  ·  boost {cfg['boost_mana']} mana"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Status", {"running": "running", "done": "finished",
                     "stopped": "stopped", "error": "error"}[state["status"]])
c2.metric("Round", f"{state.get('round', 0)} / {state.get('rounds_planned', '?')}")
c3.metric("Policies alive", f"{len(state.get('alive', []))} / {len(state['me'])}")
c4.metric("Duels run", f"{state.get('duels', 0):,}")

done, total = state.get("done", 0), max(1, state.get("total", 1))
st.progress(min(1.0, done / total))
st.caption(
    f"round {state.get('round', 0)}: cell {done}/{total} at "
    f"{state.get('round_trials', 0):,} duels each  ·  elapsed "
    f"{fmt_secs(state.get('elapsed'))}  ·  {state.get('rate', 0):,.0f} duels/sec"
)

if running and st.button("Stop this run"):
    try:
        os.kill(state["pid"], signal.SIGTERM)
        st.success("Stop signal sent")
        time.sleep(1.5)
        st.rerun()
    except OSError as e:
        st.error(f"Could not stop: {e}")

if state.get("error"):
    st.error("Solver failed")
    st.code(state["error"])

# ---------------------------------------------------------------- best ----
if state.get("best"):
    b = state["best"]
    st.success(f"**Best policy: `{b['policy']}` — {b['floor']:.1f}%** guaranteed "
               f"against their best reply (`{b['enemy_reply']}`)")
    cell = b["cell"]
    d = st.columns(4)
    d[0].metric("Win rate", f"{cell['wr']:.1f}%",
                help=f"95% CI {cell['lo']:.1f}–{cell['hi']:.1f}% over {cell['n']:,} duels")
    d[1].metric("Going first", f"{cell['first']:.1f}%")
    d[2].metric("Going second", f"{cell['second']:.1f}%")
    d[3].metric("Game length", f"{cell['rounds']:.1f} rounds")

# -------------------------------------------------------------- matrix ----
st.subheader("Win rate %, rows = my policy, columns = theirs")
alive = set(state.get("alive", []))
elim = state.get("eliminated", {})
rows = [r for r in state["me"] if r in alive] + \
       [r for r in state["me"] if r not in alive]
data = {c: [state["cells"].get(f"{r}|{c}", {}).get("wr") for r in rows]
        for c in state["foe"]}
# cast explicitly: a column with no results yet is all-None, which pandas keeps
# as object dtype and then refuses to compare
df = pd.DataFrame(data, index=rows).astype("float64")
df["FLOOR"] = df[state["foe"]].min(axis=1, skipna=False)
df["duels"] = [state["cells"].get(f"{r}|{state['foe'][0]}", {}).get("n")
               for r in rows]
df.index = [r if r in alive else f"{r}  (out r{elim.get(r, {}).get('round', '?')})"
            for r in rows]
st.dataframe(
    df.style.map(shade, subset=state["foe"] + ["FLOOR"])
      .format(cellfmt, subset=state["foe"] + ["FLOOR"])
      .format("{:,.0f}", subset=["duels"], na_rep="·"),
    use_container_width=True)
st.caption(
    "FLOOR is the worst column — the win rate the policy guarantees against "
    "their best reply. Policies are eliminated when their best possible floor "
    "falls below the leader's worst possible floor, so the surviving rows get "
    "all the remaining duels."
)

if elim:
    st.subheader("Eliminated")
    st.dataframe(pd.DataFrame([
        {"policy": k, "round": v["round"], "floor %": round(v["floor"], 1),
         "best case %": round(v["ceiling"], 1), "leader's bar %": round(v["bar"], 1),
         "duels/cell": v["n"]} for k, v in
        sorted(elim.items(), key=lambda x: x[1]["round"])],
    ), use_container_width=True, hide_index=True)

# --------------------------------------------------------- head to head ----
st.subheader("Head to head")

run_cfg = {"grid.BERRY_WEIGHT": cfg["berry_rate"],
           "MANA_CARRYOVER": cfg["mana_carryover"],
           "BASE_HP": cfg["base_hp"]}


@st.cache_data(show_spinner=False)
def h2h(my_pol, foe_pol, n, cfg_key):
    import json
    return analysis.head_to_head(my_pol, foe_pol, n, cfg=json.loads(cfg_key))


defaults = state.get("best") or {}
mine = state["me"]
theirs = state["foe"]
h1, h2, h3, h4 = st.columns([3, 3, 2, 2])
pick_me = h1.selectbox("My policy", mine,
                       index=mine.index(defaults.get("policy", mine[0]))
                       if defaults.get("policy") in mine else 0)
pick_foe = h2.selectbox("Their policy", theirs,
                        index=theirs.index(defaults.get("enemy_reply", theirs[0]))
                        if defaults.get("enemy_reply") in theirs else 0)
n_h2h = h3.selectbox("Duels", [1000, 2500, 5000, 10000], index=1)
go = h4.button("Simulate", type="primary")

if go:
    st.session_state["h2h"] = (pick_me, pick_foe, n_h2h)

if st.session_state.get("h2h"):
    m, f, n = st.session_state["h2h"]
    import json
    with st.spinner(f"running {n:,} duels of {m} vs {f}…"):
        d = h2h(m, f, n, json.dumps(run_cfg, sort_keys=True))
    k = st.columns(4)
    k[0].metric("My win rate", f"{d['win']:.1f}%",
                help=f"95% CI {d['lo']:.1f}–{d['hi']:.1f}% over {d['n']:,} duels")
    k[1].metric("Going first", f"{d['first']:.1f}%")
    k[2].metric("Going second", f"{d['second']:.1f}%")
    k[3].metric("Game length", f"{d['rounds']:.1f} rounds")

    st.markdown("**Mana budget per game** — where the income actually goes")
    st.dataframe(pd.DataFrame([
        {"team": "mine", **{kk: round(vv, 1) for kk, vv in d["me"].items()}},
        {"team": "theirs", **{kk: round(vv, 1) for kk, vv in d["foe"].items()}},
    ]).rename(columns={"earned": "mana earned", "drained": "lost to drain",
                       "wasted": "burnt overshooting", "spent": "mana spent",
                       "usable_pct": "% of income used",
                       "dmg_per_mana": "damage per mana",
                       "setups": "repositions/game", "ruin": "ruin matches/game",
                       "extra": "extra moves/game", "berries": "berries/game"}),
        use_container_width=True, hide_index=True)

    st.markdown("**Per monster**")
    st.dataframe(pd.DataFrame(d["monsters"]).round(2),
                 use_container_width=True, hide_index=True)

    c = st.columns(2)
    c[0].markdown(f"**`{m}`**\n\n" + "\n\n".join(
        "- " + b for b in ai.describe(ai.MY_POLICIES[m])))
    c[1].markdown(f"**`{f}`**\n\n" + "\n\n".join(
        "- " + b for b in ai.describe(ai.FOE_POLICIES[f])))

# ------------------------------------------------------------ top table ----
done_rows = [r for r in state["me"]
             if all(f"{r}|{c}" in state["cells"] for c in state["foe"])]
if len(done_rows) >= 2:
    st.subheader("Best against best")
    topme = sorted(done_rows,
                   key=lambda r: -min(state["cells"][f"{r}|{c}"]["wr"]
                                      for c in state["foe"]))[:4]
    topfoe = sorted(state["foe"],
                    key=lambda c: max(state["cells"][f"{r}|{c}"]["wr"]
                                      for r in done_rows))[:4]
    sub = pd.DataFrame(
        {c: [state["cells"][f"{r}|{c}"]["wr"] for r in topme] for c in topfoe},
        index=topme).astype("float64")
    sub["FLOOR"] = sub[topfoe].min(axis=1)
    st.dataframe(sub.style.map(shade).format(cellfmt), use_container_width=True)
    st.caption("The four policies with the best floors against the four enemy "
               "replies that hurt most — the corner of the matrix that decides "
               "the matchup.")

with st.expander("What every policy does"):
    st.markdown("#### My policies")
    for p in state["me"]:
        st.markdown(f"**`{p}`**  \n" + "  \n".join(
            "· " + b for b in ai.describe(ai.MY_POLICIES[p])))
    st.markdown("#### Their policies")
    for p in state["foe"]:
        st.markdown(f"**`{p}`**  \n" + "  \n".join(
            "· " + b for b in ai.describe(ai.FOE_POLICIES[p])))

with st.expander("Per-cell detail"):
    rowsd = []
    for r in state["me"]:
        for c in state["foe"]:
            d = state["cells"].get(f"{r}|{c}")
            if d:
                rowsd.append({"my policy": r, "their policy": c,
                              "win %": round(d["wr"], 1),
                              "95% CI": f"{d['lo']:.1f}–{d['hi']:.1f}",
                              "first %": round(d["first"], 1) if d["first"] else None,
                              "second %": round(d["second"], 1) if d["second"] else None,
                              "rounds": round(d["rounds"], 1), "duels": d["n"]})
    if rowsd:
        st.dataframe(pd.DataFrame(rowsd), use_container_width=True, hide_index=True)

with st.expander("Solver log"):
    try:
        with open(progress.log_path(sel)) as f:
            st.code(f.read()[-4000:] or "(empty)")
    except OSError:
        st.write("no log yet")

if auto and running:
    time.sleep(every)
    st.rerun()
