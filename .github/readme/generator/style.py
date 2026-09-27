"""Restrained scientific styling for the README graphics."""
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image

# Quiet, print-oriented palette shared by all README graphics.
PAPER = "#FFFFFF"
INK = "#273442"
SECONDARY = "#526170"
MUTED = "#65717C"
GRID = "#E1E5E8"
LABEL = "#526170"
BLUE = "#4D708A"
RUST = "#A47758"
LIGHT = "#F3F5F6"

# Render at native UHD resolution; never enlarge a low-resolution raster.
ANIMATION_FIGSIZE = (8, 4.5)
ANIMATION_DPI = 480

# Monotonic light-to-dark ramps: larger values have more visual weight.
SOLUTION_CMAP = LinearSegmentedColormap.from_list("solution", ["#E5EBEF", "#98AFBE", "#4D708A"])
ERROR_CMAP = LinearSegmentedColormap.from_list("error", ["#F5F2ED", "#C8AC94", "#916647"])

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


def save_gif(frames, path, durations, colors=256):
    """Write a UHD GIF with a stable full palette and a lossless final still.

    Every frame contributes to the palette. Small palette-training previews keep
    memory bounded without reducing the resolution of the encoded frames.
    """
    thumb_size = (480, 270)
    sample = Image.new("RGB", (thumb_size[0], thumb_size[1] * len(frames)))
    for k, frame in enumerate(frames):
        sample.paste(frame.resize(thumb_size, Image.Resampling.LANCZOS),
                     (0, k * thumb_size[1]))
    palette = sample.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    # Preserve the exact white page background after adaptive quantisation.
    entries = palette.getpalette()
    whitest = max(range(colors), key=lambda i: sum(entries[3 * i:3 * i + 3]))
    entries[3 * whitest:3 * whitest + 3] = [255, 255, 255]
    palette.putpalette(entries)
    # Dithering preserves subtle surface gradients within GIF's colour limit.
    quantised = [frame.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG)
                 for frame in frames]
    quantised[0].save(path, save_all=True, append_images=quantised[1:], duration=durations,
                      loop=0, optimize=False, disposal=1)
    path = Path(path)
    frames[-1].save(path.with_name(path.stem + "_still.png"), optimize=True)


def style_axes(ax):
    ax.set_facecolor(PAPER)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.7)
    ax.tick_params(colors=MUTED, labelsize=9, length=0, pad=4)
    ax.grid(True, color=GRID, linewidth=0.5, linestyle="-")
    ax.set_axisbelow(True)
