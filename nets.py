"""Actor-critic for both agents. Width and depth are configurable so the
capacity question can be tested rather than guessed at."""
import torch
import torch.nn as nn

import grid
from selfplay import N_ACTIONS, N_PLANES, SCALARS


def pick_device(prefer=None):
    """cuda on Colab, mps on Apple silicon, cpu otherwise."""
    if prefer and prefer != 'auto':
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n1 = nn.GroupNorm(8, ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n2 = nn.GroupNorm(8, ch)
        self.act = nn.ReLU()

    def forward(self, x):
        y = self.act(self.n1(self.c1(x)))
        y = self.n2(self.c2(y))
        return self.act(x + y)


class ActorCritic(nn.Module):
    """A small residual CNN over the board plus an MLP over the scalars.

    The board is only 5x7, so depth matters more than width -- a match is a
    local pattern, but whether it is worth making depends on mana bars that
    live in the scalar vector, so the trunk has to mix the two.
    """

    def __init__(self, width=128, blocks=6, hidden=512):
        super().__init__()
        self.width, self.blocks, self.hidden = width, blocks, hidden
        self.stem = nn.Sequential(
            nn.Conv2d(N_PLANES, width, 3, padding=1),
            nn.GroupNorm(8, width), nn.ReLU())
        self.body = nn.Sequential(*[ResBlock(width) for _ in range(blocks)])
        self.scal = nn.Sequential(
            nn.Linear(SCALARS, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU())
        self.trunk = nn.Sequential(
            nn.Linear(width * grid.H * grid.W + 128, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.pi = nn.Linear(hidden, N_ACTIONS)
        self.v = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(),
                               nn.Linear(128, 1))

    def forward(self, board, scalars, mask):
        x = self.body(self.stem(board)).flatten(1)
        h = self.trunk(torch.cat([x, self.scal(scalars)], 1))
        logits = self.pi(h).masked_fill(mask == 0, float('-inf'))
        return logits, self.v(h).squeeze(-1)

    @property
    def arch(self):
        return {'width': self.width, 'blocks': self.blocks, 'hidden': self.hidden}
