"""2-D target potentials used for the Figure 4 density-matching reproduction.

These are the standard U1-U4 toy targets from Rezende & Mohamed (2015), the
normalizing-flow setup cited by Chen et al. for this experiment.

Which three are in Neural ODE Figure 4?  Verified by rendering the attached PDF page 5
at 300 dpi and comparing the target column with every candidate:
  row 1: two crescents "( )"                       -> U1 (ring x two x-modes)
  row 2: sine band with a bump branch that rejoins  -> U3 (second mode w1 - w2, w2 = bump)
  row 3: sine band that permanently splits at z1~1  -> U4 (second mode w1 - w3, w3 = sigmoid step)
U2 (a single plain sine band) does NOT appear in the paper figure.  The previous
version of this repo used (U1, U3, U2); row 3 was wrong and is now U4.

Orientation: the sine in the paper's target images has its crest at z1 = -1 and its
trough at z1 = +1, i.e. the paper images are the math-orientation density flipped
vertically (matplotlib imshow default origin='upper' with an ascending y-grid).
The branches therefore appear *above* the main band although w1 - w2 and w1 - w3 lie
below it.  figure4 reproduces this orientation (PAPER_IMAGE_FLIPPED = True).
"""

import math
import torch


def w1(z):
    return torch.sin(2.0 * math.pi * z[:, 0] / 4.0)


def w2(z):
    return 3.0 * torch.exp(-0.5 * ((z[:, 0] - 1.0) / 0.6) ** 2)


def w3(z):
    return 3.0 * torch.sigmoid((z[:, 0] - 1.0) / 0.3)


def U1(z):
    r = torch.linalg.norm(z, dim=1)
    a = 0.5 * ((r - 2.0) / 0.4) ** 2
    b = torch.exp(-0.5 * ((z[:, 0] - 2.0) / 0.6) ** 2)
    c = torch.exp(-0.5 * ((z[:, 0] + 2.0) / 0.6) ** 2)
    return a - torch.log(b + c + 1e-12)


def U2(z):
    return 0.5 * ((z[:, 1] - w1(z)) / 0.4) ** 2


def U3(z):
    a = torch.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.35) ** 2)
    b = torch.exp(-0.5 * ((z[:, 1] - w1(z) + w2(z)) / 0.35) ** 2)
    return -torch.log(a + b + 1e-12)


def U4(z):
    a = torch.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.4) ** 2)
    b = torch.exp(-0.5 * ((z[:, 1] - w1(z) + w3(z)) / 0.35) ** 2)
    return -torch.log(a + b + 1e-12)


TARGETS = {"u1": U1, "u2": U2, "u3": U3, "u4": U4}
PAPER_FIGURE4_ROWS = ("u1", "u3", "u4")
PAPER_IMAGE_FLIPPED = True
