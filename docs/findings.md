# Match Monsters Duel Simulator

Simulates **Bonzumi (Fire) + Sipzap (Electric)** against **Pelijet (Water) +
Barbenin (Psychic)** on a real 7x5 board, to answer whether the team is good
enough rather than guessing from ability text.

| File | What it is |
|---|---|
| `grid.py` | Board engine: tiles, swaps, gravity, refill, cascades, ability effects |
| `monsters.py` | Monster stats and abilities, in one editable place |
| `sim.py` | Duel engine, the AI, and all the reports |
| `duel_v3_legacy.py`, `grid.py.bak` | The previous version, kept for comparison |

```bash
python3 sim.py            # everything
```

Individual reports: `python3 sim.py board | matrix | head | sens | ai`.

---

## Rules the model implements

Marked **[C]** where confirmed — by the official Plato rules page, the official
monster descriptions, or by you checking in game. **[A]** marks a genuine
assumption; every one of those is measured in the sensitivity report.

### Board

- **[C]** 7 columns x 5 rows.
- **[C]** Six tile types — red, yellow, blue, green, purple, berry — at **the
  same spawn rate, 1/6 each**.
- **[C]** A batch of new tiles never arrives already matched *among itself*, but
  a falling tile **may** land completing a match with tiles already settled.
  That match then resolves, so **cascades happen**.
- **[C]** Matches are horizontal or vertical, 3 or more.
- **[C]** A swap does **not** have to create a match. You may spend a move just
  repositioning a tile, and take a second move to complete the match — so unlike
  Candy Crush there is no such thing as a stuck board. If nothing is matchable
  you shuffle a tile and carry on.

### Turns

- **[C]** 2 moves per turn. A match of 4+ grants **one** extra move per turn
  (Normal mode; Match Bonus mode allows more, measured in `sens`).
- **[C]** Base HP is 80, and whoever goes **second gets +5 HP**. So it is 80 vs
  85, not a random roll.
- **[C]** A full mana bar fires immediately and automatically.

### Mana and berries

- **[C]** Matching a colour charges *your* monster of that colour. Mana is
  private; the shared resource is the board.
- **[C]** Matching a colour neither of your monsters uses does nothing.
- **[C]** Berries are a single team counter that **holds at most 4**. At 4 you
  may evolve **either** monster — your choice — and the counter resets to 0.
  Berries collected past 4 are wasted.
- **[C]** Evolving **costs one of your two moves**.
- **[C]** An already-evolved monster fed 4 berries gets a mana boost instead.
- **[A]** That boost is worth 4 mana.
- **[A]** Overflow mana carries over rather than being zeroed.

## The monsters

### Mine

**Bonzumi** — Fire, 8 mana. *Flare*: 20 damage and **matches a random column**
— you do not get to pick it. Every red tile in that column feeds Bonzumi, every
yellow tile feeds Sipzap, every berry bumps the berry counter, and anything else
is simply removed. Evolves to **Bonzire**: 25 damage, two columns.

**Sipzap** — Electric, 4 mana. *Honey Ohm*, driven by a honey charge:

| Charge | Damage | Tiles collected |
|---|---|---|
| 0 | 0 | 0 |
| 1 | 5 | 2 |
| 2 | 10 | 4 |
| 3 *(Ranzap only)* | 20 | 8 |

The charge rises by 1 at the **end of every turn no matter what happens**, and
**resets to 0 the instant Sipzap fires**. The collected tiles are pulled at
random off the board and routed exactly like Flare's column. Evolving to
**Ranzap** raises the cap from 2 to 3.

The consequence, which the AI models explicitly: **filling Sipzap's bar while
its charge is 0 throws the mana away.** The move scorer gives that a *negative*
score rather than pretending mana is mana.

### Theirs

**Pelijet** — Water, 6 mana. 10 damage, converts 3 random tiles to Water.
Evolved: 20 damage. The conversion is self-funding — it keeps replacing the blue
tiles it just spent.

**Barbenin** — Psychic, 6 mana. 10 damage and drains 2 mana from **each** of my
monsters. Evolved: 15 damage, drains 3 from each.

Green belongs to nobody in this matchup, so matching it is pure tempo loss.

---

## How the AI plays

Both sides run the same engine, so the comparison is fair. A move is scored in
**damage-equivalent units**, not tile counts:

- Each monster's per-activation value is solved as a fixed point, because
  abilities feed each other (Flare's column refills Bonzumi and Sipzap; Honey
  Ohm's collected tiles refill both).
- A tile is worth that activation value divided by the mana cost — so an 8-cost
  Bonzumi tile and a 4-cost Sipzap tile are priced correctly against each other.
- Mana is valued against the **threshold**, not linearly. Crossing the line to
  fire *now* earns a tempo bonus; crossing it when Sipzap's charge is 0 scores
  negative.
- Damage past the opponent's remaining HP is capped, so the AI does not
  over-value overkill when the enemy is nearly dead.
- Berries are priced as a quarter of an evolution, minus the move that
  evolution costs.
- A 4+ match is credited with the extra move it earns.
- **Repositioning:** the AI compares two-move plans. It spends a move moving a
  tile only when the match that unlocks beats the best match available right now
  *plus* what an ordinary next move would have been worth.

Denial (matching enemy colours) is a weight, not a hard rule, and the strategy
matrix sweeps it rather than assuming a value.

## Findings

Numbers below are from `sim.py` on the corrected rules. Confidence intervals are
Wilson 95%. Both sides run the same AI, so the comparison is fair.

### 0. The headline: 51.2%

Against the enemy's best reply, playing my own best plan, over 12,000 duels:

| My plan | Floor vs their best reply | 95% CI |
|---|---|---|
| **bon_hdeny** (evolve Bonzumi, heavy denial) | **51.2%** | 50.3 - 52.1 |
| bon_deny (evolve Bonzumi, moderate denial) | 50.6% | 49.7 - 51.5 |
| bon_evo (evolve Bonzumi, no denial) | 42.6% | 41.8 - 43.5 |

The enemy's best reply in every case is `pel_deny` -- evolve Pelijet and deny in
moderation. Full row, my win % by their plan:

| | pel_evo | bar_evo | pel_deny | bar_deny | pel_hdeny | no_berry |
|---|---|---|---|---|---|---|
| bon_hdeny | 58.6 | 67.0 | **51.2** | 57.1 | 54.1 | 82.9 |

**The team is a coin flip, very slightly favoured.** It is not the losing team an
earlier version of this model reported.

### 1. Bonzumi's 8-cost is why it isn't better than even

Bonzumi's `Flare` costs 8. Both enemy monsters cost 6. Two moves a turn earns
roughly 6 mana, so Pelijet and Barbenin fire on rhythm while Bonzumi is
permanently two short -- and Barbenin's drain steals exactly that much.

| Change | Win % | Delta |
|---|---|---|
| Bonzumi costs 6 | 82.3 | **+26.0** |
| Bonzumi costs 7 | 69.4 | **+13.1** |
| Barbenin does not drain at all | 64.6 | +8.3 |
| baseline | 56.3 | -- |

Cutting Bonzumi's cost by 2 is worth more than deleting Barbenin's ability
outright. **This is a roster problem, not a piloting problem** -- and it is the
one conclusion that has survived every correction to the model.

### 2. Evolve Bonzumi, never Sipzap

Worth about 25 points of win rate. Ranzap's charge-3 tier (20 damage) needs
three quiet turns to reach, but the mana bar keeps filling and Sipzap fires
automatically, so the higher cap is mostly wasted. Bonzire's second column is
available every time it fires.

### 3. You cannot skip the berry race

Removing evolution entirely costs 11.7 points. At a 1/6 spawn rate a berry match
is available about 50% of the time, so Pelijet **will** evolve from 10 to 20
damage. Ignoring berries is the worst plan tested by a wide margin.

An earlier version of this model used a 6% berry rate and concluded berries were
unmatchable and evolution never happened. That was an artifact of the wrong
spawn rate; it is not true.

### 4. Repositioning is a rescue, not a strategy

Holding everything else fixed and varying only how dearly the AI prices a
repositioning move:

| Repositioning threshold | Moves spent repositioning | Win % |
|---|---|---|
| eager | 17% | 51.4 |
| moderate | 9% | 56.3 |
| strict | 4% | 56.1 |
| very strict | 2% | 56.7 |
| never (forced only) | 2% | 56.4 |

Everything from "strict" upwards is within noise of not doing it at all. A 4+
match only refunds the move you spent setting it up, so building one
deliberately breaks even at best. The mechanic earns its keep in the ~2% of
moves where nothing is matchable. `setup_cost_w` defaults to 2.0 for this
reason; at 1.0 both sides play measurably worse.

### 5. Going first is worth about 7 points, and +5 HP does not pay for it

Games last around 8-9 rounds each, so the tempo of an extra strike outweighs the
second player's HP bonus.

### 6. Assumptions that turned out not to matter

Cascades (+0.7), abilities re-triggering (-0.0), and Match Bonus mode (-0.1) are
all within noise. The assumptions still worth checking in game are mana
carryover (-4.2 if wrong) and the size of the berry boost (-3.9 if it is 2 mana
rather than 4).

---

## What is still not modelled

- **The timer.** 30 seconds a move; no time pressure exists in the sim.
- **Stages/anomalies.** Stadium (60 HP), Forest Valley (3 berries), Lava
  Caverns, etc. all exist in the real game and are not implemented.
- **Deep planning.** The AI looks one move ahead for setups. It does not plan a
  three-move chain or deliberately leave the board barren for the opponent. A
  strong human is better than this — hopefully symmetrically, since both sides
  use the same brain.
