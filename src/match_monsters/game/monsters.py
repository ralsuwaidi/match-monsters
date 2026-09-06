"""
Monster definitions. Data-driven so numbers can be corrected in one place.

Stats for the four monsters in this matchup come from the user's in-game
screenshots. Ability shapes (damage / convert / clear / drain / heal / extra
moves) follow the vocabulary Plato uses on the official Match Monsters page.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Ability:
    damage: int = 0
    heal: int = 0
    drain: int = 0                      # mana removed from EACH enemy monster
    convert_n: int = 0
    convert_color: Optional[str] = None
    collect_n: int = 0                  # pull N random tiles off the board
    clear_columns: int = 0
    clear_rows: int = 0
    extra_moves: int = 0


@dataclass(frozen=True)
class Monster:
    name: str
    color: str
    cost: int
    base: Optional[Ability] = None
    evo: Optional[Ability] = None
    # Charge-based monsters (Yellow): ability depends on charges accumulated.
    charge_base: Optional[Tuple[Ability, ...]] = None
    charge_evo: Optional[Tuple[Ability, ...]] = None
    charge_cap: int = 0
    charge_cap_evo: int = 0
    charges_per_turn: int = 0

    @property
    def charged(self) -> bool:
        return self.charge_base is not None

    def cap(self, evolved: bool) -> int:
        if not self.charged:
            return 0
        return self.charge_cap_evo if evolved else self.charge_cap

    def ability(self, evolved: bool, charges: int = 0) -> Ability:
        if self.charged:
            table = self.charge_evo if evolved else self.charge_base
            return table[min(charges, self.cap(evolved))]
        return self.evo if evolved else self.base


_Y = 'yellow'


def honey_ohm(cap, base_dmg=5, base_tiles=2, zero_dead=False):
    """Sipzap / Ranzap. Official wording: "attacks for 5 HP and collects 2
    random tiles -- each value is doubled for every charge of honey".

    `zero_dead` switches to the alternative reading (from the in-game
    screenshots) where 0 charges does nothing and charge 1 is the 5/2 tier."""
    tab = []
    for ch in range(cap + 1):
        if zero_dead and ch == 0:
            tab.append(Ability())
            continue
        mult = 2 ** (ch - 1) if zero_dead else 2 ** ch
        tab.append(Ability(damage=base_dmg * mult, collect_n=base_tiles * mult))
    return tuple(tab)


# [C] user-confirmed in game: 0 charges does nothing, charge 1 = 5 dmg / 2 tiles,
# charge 2 = 10 dmg / 4 tiles, and evolved Ranzap adds charge 3 = 20 dmg / 8 tiles.
def honey_ohm_convert(cap, base_dmg=5, base_tiles=2, zero_dead=True):
    """Alternative reading: the tile number is tiles CONVERTED to yellow on the
    board rather than collected off it. Swept in the sensitivity report."""
    tab = []
    for ch in range(cap + 1):
        if zero_dead and ch == 0:
            tab.append(Ability())
            continue
        mult = 2 ** (ch - 1) if zero_dead else 2 ** ch
        tab.append(Ability(damage=base_dmg * mult,
                           convert_n=base_tiles * mult, convert_color=_Y))
    return tuple(tab)


SIPZAP_ZERO_DEAD = True

# Fire. "Flare: attacks for 20 HP and matches a random column (requires 8 Fire)."
# Evolves into Bonzire.
BONZUMI = Monster(
    name='Bonzumi', color='red', cost=8,
    base=Ability(damage=20, clear_columns=1),
    evo=Ability(damage=25, clear_columns=2),
)

# Electric. "Honey Ohm: attacks for 5 HP and collects 2 random tiles -- each
# value doubled per charge of honey, limit 2 charges." Evolves into Ranzap
# (limit 3 charges).
#
# [C] user-confirmed: the charge rises by 1 at the END of every turn no matter
# what happens, and resets to 0 the moment Sipzap fires -- so filling its bar
# while the charge is 0 wastes the mana entirely. The 2/4/8 tiles are pulled at
# random off the board: red/yellow ones feed that monster's mana bar, berries
# increment the berry counter, enemy colours are simply removed.
SIPZAP = Monster(
    name='Sipzap', color='yellow', cost=4,
    charge_base=honey_ohm(2, zero_dead=SIPZAP_ZERO_DEAD),
    charge_evo=honey_ohm(3, zero_dead=SIPZAP_ZERO_DEAD),
    charge_cap=2, charge_cap_evo=3, charges_per_turn=1,
)

# Water. "Hydro Rush: 10 HP + converts 3 random tiles to Water."
PELIJET = Monster(
    name='Pelijet', color='blue', cost=6,
    base=Ability(damage=10, convert_n=3, convert_color='blue'),
    evo=Ability(damage=20, convert_n=3, convert_color='blue'),
)

# Psychic. "Psycho Bite: attacks for 10 HP and drains 2 Mana from opponent's
# monsters (requires 6 Psychic)."
BARBENIN = Monster(
    name='Barbenin', color='purple', cost=6,
    base=Ability(damage=10, drain=2),
    evo=Ability(damage=15, drain=3),
)

# kept as aliases so older references still resolve
RED, YELLOW = BONZUMI, SIPZAP

MY_TEAM = (BONZUMI, SIPZAP)
FOE_TEAM = (PELIJET, BARBENIN)
