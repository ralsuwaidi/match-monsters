"""
Match Monsters board engine.

7 wide x 5 tall. A move = swap two orthogonally adjacent tiles; legal only if
the swap creates a run of 3+. Matched tiles vanish, columns fall, new tiles
drop in from the top.

CONFIRMED RULES
  * 5 colours + berries, all with the SAME spawn rate (1/6 each).
  * Newly spawned tiles never appear already matched.
  * Tiles that FALL can land in a match -> cascades resolve.
"""

import random

W, H = 7, 5
COLORS = ['red', 'yellow', 'blue', 'green', 'purple']
BERRY = 'berry'
ALL = COLORS + [BERRY]

# [C] Berries drop at 10%, rarer than the colours. The five colours split the
# remainder evenly at (1 - 0.10)/5 = 18% each.
BERRY_WEIGHT = 0.10

# Falling tiles can complete a match (user-confirmed).
CASCADE = True

# Ability-driven board changes (column clears, tile conversions) also settle
# and can trigger matches. Assumed, not confirmed.
RESOLVE_AFTER_ABILITY = True

MAX_CASCADE_STEPS = 60


def _weighted(rng):
    return BERRY if rng.random() < BERRY_WEIGHT else rng.choice(COLORS)


class Grid:
    def __init__(self, rng):
        self.rng = rng
        self.g = [[None] * W for _ in range(H)]
        self.last_longest = 0   # longest run in the most recent ability resolve
        self._fill()

    # ---------- basic ----------
    def at(self, r, c):
        return self.g[r][c]

    def counts(self):
        d = {}
        for row in self.g:
            for t in row:
                d[t] = d.get(t, 0) + 1
        return d

    def _would_make_run(self, r, c, color, new_only=None):
        """Would placing `color` at (r,c) create a 3-run?

        `new_only` restricts the check to cells filled in THIS refill pass. That
        is the real rule: a batch of new tiles never arrives already matched,
        but a new tile IS allowed to land completing a match with tiles that
        were already on the board -- which then resolves as a cascade."""
        def same(rr, cc):
            if self.g[rr][cc] != color:
                return False
            return new_only is None or (rr, cc) in new_only

        n = 1
        cc = c - 1
        while cc >= 0 and same(r, cc):
            n += 1; cc -= 1
        cc = c + 1
        while cc < W and same(r, cc):
            n += 1; cc += 1
        if n >= 3:
            return True
        n = 1
        rr = r - 1
        while rr >= 0 and same(rr, c):
            n += 1; rr -= 1
        rr = r + 1
        while rr < H and same(rr, c):
            n += 1; rr += 1
        return n >= 3

    def _fill(self):
        """Fill empty cells top-down. A batch of new tiles never arrives
        already matched among itself; matching with tiles that were already
        settled is allowed and resolves as a cascade."""
        new = set()
        for r in range(H):
            for c in range(W):
                if self.g[r][c] is not None:
                    continue
                placed = False
                for _ in range(24):
                    col = _weighted(self.rng)
                    if not self._would_make_run(r, c, col, new):
                        self.g[r][c] = col
                        placed = True
                        break
                if not placed:
                    for col in ALL:
                        if not self._would_make_run(r, c, col, new):
                            self.g[r][c] = col
                            placed = True
                            break
                if not placed:
                    self.g[r][c] = self.rng.choice(COLORS)
                new.add((r, c))

    def _gravity(self):
        for c in range(W):
            col = [self.g[r][c] for r in range(H) if self.g[r][c] is not None]
            newcol = [None] * (H - len(col)) + col
            for r in range(H):
                self.g[r][c] = newcol[r]

    # ---------- matching ----------
    def find_runs(self):
        """All (r,c) inside a run of 3+, plus the longest run length."""
        hits = set()
        longest = 0
        for r in range(H):
            c = 0
            while c < W:
                c2 = c
                while c2 + 1 < W and self.g[r][c2 + 1] == self.g[r][c]:
                    c2 += 1
                if c2 - c + 1 >= 3:
                    longest = max(longest, c2 - c + 1)
                    for x in range(c, c2 + 1):
                        hits.add((r, x))
                c = c2 + 1
        for c in range(W):
            r = 0
            while r < H:
                r2 = r
                while r2 + 1 < H and self.g[r2 + 1][c] == self.g[r][c]:
                    r2 += 1
                if r2 - r + 1 >= 3:
                    longest = max(longest, r2 - r + 1)
                    for x in range(r, r2 + 1):
                        hits.add((x, c))
                r = r2 + 1
        return hits, longest

    def _runs_through(self, r, c, hits, longest):
        """Runs of 3+ passing through (r,c). Accumulates into `hits`."""
        g = self.g
        col = g[r][c]
        c1 = c
        while c1 > 0 and g[r][c1 - 1] == col:
            c1 -= 1
        c2 = c
        while c2 < W - 1 and g[r][c2 + 1] == col:
            c2 += 1
        n = c2 - c1 + 1
        if n >= 3:
            if n > longest:
                longest = n
            for x in range(c1, c2 + 1):
                hits.add((r, x))
        r1 = r
        while r1 > 0 and g[r1 - 1][c] == col:
            r1 -= 1
        r2 = r
        while r2 < H - 1 and g[r2 + 1][c] == col:
            r2 += 1
        n = r2 - r1 + 1
        if n >= 3:
            if n > longest:
                longest = n
            for y in range(r1, r2 + 1):
                hits.add((y, c))
        return longest

    def _runs_near(self, cells):
        """Runs created by moving tiles into `cells`. Exact, because the board
        carries no runs before a swap."""
        hits = set()
        longest = 0
        for (r, c) in cells:
            longest = self._runs_through(r, c, hits, longest)
        return hits, longest

    def resolve(self, cascade=None):
        """Clear every run, apply gravity + refill, repeat while cascading.
        Returns (dict tile->count cleared, longest run seen)."""
        if cascade is None:
            cascade = CASCADE
        total, longest = {}, 0
        for step in range(MAX_CASCADE_STEPS):
            hits, ln = self.find_runs()
            if not hits:
                break
            longest = max(longest, ln)
            for (r, c) in hits:
                t = self.g[r][c]
                total[t] = total.get(t, 0) + 1
                self.g[r][c] = None
            self._gravity()
            self._fill()
            if not cascade:
                break
        return total, longest

    # ---------- moves ----------
    def moves_with_preview(self):
        """Every legal swap with what it immediately clears.
        -> list of ((r1,c1,r2,c2), cleared_dict, longest). Pre-cascade."""
        out = []
        g = self.g
        for r in range(H):
            for c in range(W):
                for dr, dc in ((0, 1), (1, 0)):
                    r2, c2 = r + dr, c + dc
                    if r2 >= H or c2 >= W:
                        continue
                    if g[r][c] == g[r2][c2]:
                        continue
                    g[r][c], g[r2][c2] = g[r2][c2], g[r][c]
                    hits, longest = self._runs_near(((r, c), (r2, c2)))
                    if hits:
                        got = {}
                        for (hr, hc) in hits:
                            t = g[hr][hc]
                            got[t] = got.get(t, 0) + 1
                        out.append(((r, c, r2, c2), got, longest))
                    g[r][c], g[r2][c2] = g[r2][c2], g[r][c]
        return out


    def _window_pot(self, r, c):
        """Best count of same-coloured tiles in any 3-window through (r,c).
        2 means one further swap could complete a match here."""
        g = self.g
        col = g[r][c]
        best = 1
        row = g[r]
        for start in range(max(0, c - 2), min(c, W - 3) + 1):
            n = ((row[start] == col) + (row[start + 1] == col)
                 + (row[start + 2] == col))
            if n > best:
                best = n
        for start in range(max(0, r - 2), min(r, H - 3) + 1):
            n = ((g[start][c] == col) + (g[start + 1][c] == col)
                 + (g[start + 2][c] == col))
            if n > best:
                best = n
        return best

    def all_swaps(self, weights=None, want_setups=True):
        """Every legal adjacent swap, split into matches and setups.

        Non-matching swaps are legal in Match Monsters (unlike Candy Crush):
        you may spend a move repositioning a tile, then a second move to
        complete the match. Those are returned as `setups`, ranked by how
        promising they look, so a caller can search only the best few.

        -> (matches: [(mv, cleared, longest)], setups: [(mv, potential)])"""
        matches, setups = [], []
        g = self.g
        for r in range(H):
            for c in range(W):
                for dr, dc in ((0, 1), (1, 0)):
                    r2, c2 = r + dr, c + dc
                    if r2 >= H or c2 >= W:
                        continue
                    a, b = g[r][c], g[r2][c2]
                    if a == b:
                        continue
                    g[r][c], g[r2][c2] = b, a
                    hits, longest = self._runs_near(((r, c), (r2, c2)))
                    if hits:
                        got = {}
                        for (hr, hc) in hits:
                            t = g[hr][hc]
                            got[t] = got.get(t, 0) + 1
                        matches.append(((r, c, r2, c2), got, longest))
                    elif want_setups:
                        p1 = self._window_pot(r, c)
                        p2 = self._window_pot(r2, c2)
                        if p1 >= 2 or p2 >= 2:
                            if weights is None:
                                pot = max(p1, p2)
                            else:
                                pot = max(p1 * weights.get(b, 0.0),
                                          p2 * weights.get(a, 0.0))
                            if pot > 0:
                                setups.append(((r, c, r2, c2), pot))
                    g[r][c], g[r2][c2] = a, b
        setups.sort(key=lambda x: -x[1])
        return matches, setups

    def any_swap(self):
        """A random legal repositioning. There is no reshuffle in this game --
        if nothing is matchable you just move tiles around."""
        opts = []
        for r in range(H):
            for c in range(W):
                for dr, dc in ((0, 1), (1, 0)):
                    r2, c2 = r + dr, c + dc
                    if r2 < H and c2 < W and self.g[r][c] != self.g[r2][c2]:
                        opts.append((r, c, r2, c2))
        return self.rng.choice(opts) if opts else None

    def moves_with_preview_near(self, cells):
        """Matches whose result could differ because `cells` changed.

        A setup swap alters two tiles. Any match not touching them was already
        available before the setup, and a pre-existing match can never beat the
        setup threshold -- so only these need scoring."""
        cand = set()
        for (r, c) in cells:
            for x in range(max(0, c - 2), min(W - 1, c + 2) + 1):
                cand.add((r, x))
            for y in range(max(0, r - 2), min(H - 1, r + 2) + 1):
                cand.add((y, c))
        out = []
        g = self.g
        seen = set()
        for (r, c) in cand:
            for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                r2, c2 = r + dr, c + dc
                if not (0 <= r2 < H and 0 <= c2 < W):
                    continue
                key = (r, c, r2, c2) if (r, c) < (r2, c2) else (r2, c2, r, c)
                if key in seen:
                    continue
                seen.add(key)
                a, b = g[r][c], g[r2][c2]
                if a == b:
                    continue
                g[r][c], g[r2][c2] = b, a
                hits, longest = self._runs_near(((r, c), (r2, c2)))
                if hits:
                    got = {}
                    for (hr, hc) in hits:
                        t = g[hr][hc]
                        got[t] = got.get(t, 0) + 1
                    out.append((key, got, longest))
                g[r][c], g[r2][c2] = a, b
        return out

    def swap(self, mv):
        """Reposition two tiles without resolving -- a setup move."""
        r, c, r2, c2 = mv
        self.g[r][c], self.g[r2][c2] = self.g[r2][c2], self.g[r][c]

    def has_move(self):
        return bool(self.moves_with_preview())

    def apply(self, mv):
        """Execute swap and settle the board. -> (cleared dict, longest run)."""
        r, c, r2, c2 = mv
        self.g[r][c], self.g[r2][c2] = self.g[r2][c2], self.g[r][c]
        return self.resolve()

    def reshuffle(self):
        """Deadlock recovery: re-deal until a move exists and nothing is matched."""
        tiles = [t for row in self.g for t in row]
        for _ in range(200):
            self.rng.shuffle(tiles)
            i = 0
            for r in range(H):
                for c in range(W):
                    self.g[r][c] = tiles[i]; i += 1
            hits, _ = self.find_runs()
            if not hits and self.has_move():
                return True
        return False

    # ---------- ability effects ----------
    def clear_cells(self, cells):
        got = {}
        for (r, c) in cells:
            t = self.g[r][c]
            if t is not None:
                got[t] = got.get(t, 0) + 1
                self.g[r][c] = None
        self._gravity()
        self._fill()
        self.last_longest = 0
        if RESOLVE_AFTER_ABILITY:
            extra, ln = self.resolve()
            self.last_longest = ln
            for t, n in extra.items():
                got[t] = got.get(t, 0) + n
        return got

    def clear_column(self, c):
        return self.clear_cells([(r, c) for r in range(H)])

    def clear_row(self, r):
        return self.clear_cells([(r, c) for c in range(W)])

    def best_column(self, colors):
        best, bestn = 0, -1
        for c in range(W):
            n = sum(1 for r in range(H) if self.g[r][c] in colors)
            if n > bestn:
                best, bestn = c, n
        return best

    def best_row(self, colors):
        best, bestn = 0, -1
        for r in range(H):
            n = sum(1 for c in range(W) if self.g[r][c] in colors)
            if n > bestn:
                best, bestn = r, n
        return best

    def collect_random(self, n):
        """Sipzap's `collect N random tiles`: pull N tiles off the board and
        hand them to the caster. Cascades from the gaps are included."""
        cells = [(r, c) for r in range(H) for c in range(W)]
        self.rng.shuffle(cells)
        return self.clear_cells(cells[:n])

    def convert(self, n, color):
        """Turn n random non-`color` tiles into `color`. Returns tiles cleared
        if the conversion happens to create matches."""
        cells = [(r, c) for r in range(H) for c in range(W)
                 if self.g[r][c] != color]
        self.rng.shuffle(cells)
        for (r, c) in cells[:n]:
            self.g[r][c] = color
        self.last_longest = 0
        if RESOLVE_AFTER_ABILITY:
            got, ln = self.resolve()
            self.last_longest = ln
            return got
        return {}


    def clone(self, rng=None):
        """Independent copy, for search. Gets its own RNG stream."""
        import random as _r
        new = object.__new__(Grid)
        new.g = [row[:] for row in self.g]
        new.last_longest = 0
        new.rng = rng if rng is not None else _r.Random(self.rng.random())
        return new
