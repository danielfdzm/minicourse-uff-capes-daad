"""Shared look for the README graphics: the course palette and the slides' typeface."""
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image

# Course palette (the \definecolor{FEM...} values of the Chapter 1 slides)
NAVY = "#17365D"
BLUE = "#2B6CB0"
TEAL = "#1597A5"
ORANGE = "#D7653B"
LIGHT = "#F4F7FA"
GRAY = "#5C6F82"
PALE = "#EAF7F9"

# Text tokens on the navy surface
INK_ON_DARK = "#FFFFFF"
INK2_ON_DARK = "#C9D6E3"      # secondary text
INK3_ON_DARK = "#8FA6BD"      # muted text / axis labels
GRID_ON_DARK = "#2A4D78"      # hairline grid, one step off the navy surface
KICKER = "#6CCAD3"            # FEMteal!70!white, as on the title slides

# Teal and orange distinguish series on white and navy backgrounds.
SERIES = (TEAL, ORANGE)

# Sequential ramps (one hue each); on navy, larger values are brighter.
TEAL_ON_DARK = LinearSegmentedColormap.from_list("teal_dark", ["#1E5775", TEAL, "#A9E6EA"])
ORANGE_ON_DARK = LinearSegmentedColormap.from_list("orange_dark", ["#24406A", "#8A4A43", ORANGE, "#F6C9A8"])

def latin_modern(name):
    """Path of a Latin Modern OpenType file from the TeX installation, or None."""
    try:
        out = subprocess.run(["kpsewhich", name], capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return None
    return out or None


def use_course_fonts():
    """Latin Modern Sans (the Beamer decks' font) when a TeX installation provides it."""
    found = False
    for name in ("lmsans10-regular.otf", "lmsans10-bold.otf", "lmsans10-oblique.otf"):
        path = latin_modern(name)
        if path:
            font_manager.fontManager.addfont(path)
            found = True
    family = "Latin Modern Sans" if found else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": [family, "DejaVu Sans"],
        # sans-serif math, as Beamer typesets it
        "mathtext.fontset": "custom",
        "mathtext.rm": family,
        "mathtext.it": f"{family}:italic",
        "mathtext.bf": f"{family}:bold",
        "mathtext.sf": family,
        "mathtext.fallback": "cm",
        "axes.unicode_minus": False,
    })


def fig_to_image(fig, size=None):
    """Render a figure to a PIL image, optionally downsampled for crisp anti-aliasing."""
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    if size is not None:
        img = img.resize(size, Image.LANCZOS)
    return img


def save_gif(frames, path, durations, colors=128):
    """Quantise frames to one shared adaptive palette and write a looping GIF."""
    sample = Image.new("RGB", (frames[0].width, frames[0].height * min(len(frames), 6)))
    step = max(1, len(frames) // 6)
    for k, fr in enumerate(frames[::step][:6]):
        sample.paste(fr, (0, k * frames[0].height))
    palette = sample.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    # no dithering: dithered noise defeats GIF compression
    quantised = [fr.quantize(palette=palette, dither=Image.Dither.NONE) for fr in frames]
    quantised[0].save(path, save_all=True, append_images=quantised[1:], duration=durations,
                      loop=0, optimize=True, disposal=1)


def style_dark_axes(ax):
    ax.set_facecolor(NAVY)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_ON_DARK)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK3_ON_DARK, labelsize=9, length=0, pad=4)
    ax.grid(True, color=GRID_ON_DARK, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
