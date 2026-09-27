"""Restrained scientific styling for the README graphics."""
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image, ImageColor

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

# Vivid data colours for animations; typography keeps the neutral tokens above.
PLOT_BLUE = "#2878B5"
PLOT_ORANGE = "#E87524"
PLOT_PURPLE = "#8056BD"
ANIMATION_SOLUTION_CMAP = matplotlib.colormaps["viridis"]
ANIMATION_ERROR_CMAP = matplotlib.colormaps["plasma"]

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


def fig_to_image(fig, size=None, transparent=False):
    """Render at native resolution, preserving alpha for transparent animations."""
    if transparent:
        fig.patch.set_alpha(0)
        for ax in fig.axes:
            ax.patch.set_alpha(0)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    img = Image.fromarray(rgba if transparent else rgba[..., :3])
    if size is not None:
        img = img.resize(size, Image.Resampling.LANCZOS)
    return img


def save_gif(frames, path, durations):
    """Write transparent UHD frames with a shared 255-colour data palette.

    GIF reserves the final palette entry for one-bit transparency. Restore the
    transparent background between frames so moving curves leave no trails.
    The final PNG retains full colour and smooth alpha at antialiased edges.
    """
    thumb_size = (480, 270)
    sample = Image.new("RGB", (thumb_size[0], thumb_size[1] * len(frames)), "white")
    for k, frame in enumerate(frames):
        thumb = frame.convert("RGBA").resize(thumb_size, Image.Resampling.LANCZOS)
        sample.paste(thumb, (0, k * thumb_size[1]), thumb.getchannel("A"))
    # Reserve exact text and series colours: adaptive quantisation alone can
    # merge small dark labels into a purple or blue from a large colour plot.
    fixed = [PAPER, INK, SECONDARY, MUTED, GRID, PLOT_BLUE, PLOT_ORANGE, PLOT_PURPLE]
    adaptive_colors = 255 - len(fixed)
    palette = sample.quantize(colors=adaptive_colors, method=Image.Quantize.MEDIANCUT)
    entries = palette.getpalette()[:3 * adaptive_colors]
    entries += [channel for color in fixed for channel in ImageColor.getrgb(color)]
    # Duplicate an existing colour in the reserved slot, then explicitly remap
    # that slot before assigning transparency. White plotted details stay opaque.
    entries += entries[:3]
    palette.putpalette(entries)
    quantised = []
    for frame in frames:
        rgba = frame.convert("RGBA")
        # Use straight RGB at edges: no white matte around text on other themes.
        rgb = rgba.convert("RGB")
        indexed = rgb.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG)
        pixels = np.array(indexed)
        pixels[pixels == 255] = 0
        pixels[np.asarray(rgba.getchannel("A")) < 128] = 255
        indexed = Image.fromarray(pixels)
        indexed.putpalette(entries)
        quantised.append(indexed)
    quantised[0].save(path, save_all=True, append_images=quantised[1:], duration=durations,
                      loop=0, optimize=False, disposal=2, transparency=255, background=255)
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
