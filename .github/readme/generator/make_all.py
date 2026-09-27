"""Regenerate the README graphics in .github/readme/ (run from anywhere).

    python .github/readme/generator/make_all.py              # everything
    python .github/readme/generator/make_all.py banner fem   # a subset

Targets: banner, roadmap, fem, ml, pinn, highlights. Needs numpy, scipy,
matplotlib, pillow and torch. The slides' typeface, Latin Modern Sans, is taken
from the TeX installation when one is available.

The animations rerun three computations: P1 finite elements for the L-shaped
eigenproblem (fem), a small tanh network fitted to noisy samples (ml) and a
physics-informed network for a 2D Poisson problem (pinn). The highlight strips
reuse figures from the chapter folders.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent            # .github/readme
ROOT = HERE.parents[2]       # repository root
sys.path.insert(0, str(HERE))

import banner       # noqa: E402
import gif_fem      # noqa: E402
import gif_ml       # noqa: E402
import gif_pinn     # noqa: E402
import highlights   # noqa: E402
import roadmap      # noqa: E402

TARGETS = {
    "banner": lambda: banner.main(OUT / "banner.png"),
    "roadmap": lambda: roadmap.main(OUT / "roadmap.png"),
    "fem": lambda: gif_fem.main(OUT / "fem_eigenmode.gif"),
    "ml": lambda: gif_ml.main(OUT / "ml_training.gif"),
    "pinn": lambda: gif_pinn.main(OUT / "pinn_training.gif"),
    "highlights": lambda: highlights.main(ROOT, OUT),
}

if __name__ == "__main__":
    for name in sys.argv[1:] or TARGETS:
        print("making", name, flush=True)
        TARGETS[name]()
