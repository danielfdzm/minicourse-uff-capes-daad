"""Per-chapter highlight strips: course figures on white cards, transparent between cards."""
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

from style import latin_modern

NAVY, BORDER, TEAL, ORANGE = "#17365D", "#CFD9E4", "#1597A5", "#D7653B"
W, MARGIN, GAP, CARD_H, PAD, CAPTION_H = 1800, 12, 24, 372, 14, 54

CH1 = "1. Theoretical and computational foundations of FEM"
STRIPS = {
    "chapter1_highlights.png": (NAVY, [
        (f"{CH1}/1.1. Theoretical foundations/figures/mesh3d_cube_slide.png", (0.4265, 0.0, 1.0, 1.0),   # blank gap between panels (a) and (b)
         "1.1  ·  Tetrahedral meshes of the unit cube"),
        (f"{CH1}/1.2. Computational practice with FEniCS/experiments/figures/stokes_cavity_3d.png", None,
         "1.2  ·  Stokes flow in a lid-driven cavity"),
        (f"{CH1}/1.2. Computational practice with FEniCS/experiments/figures/reaction_diffusion_system.png", None,
         "1.2  ·  Gray–Scott reaction–diffusion patterns"),
    ]),
    "chapter2_highlights.png": (TEAL, [
        ("2. Fundamentals of Machine Learning/figures/brain_to_ann.png", None,
         "From biological to artificial neurons"),
        ("2. Fundamentals of Machine Learning/figures/uat_width_demo.png", None,
         "Universal approximation with wider layers"),
        ("2. Fundamentals of Machine Learning/figures/under_over.png", None,
         "Underfitting and overfitting"),
    ]),
    "chapter3_highlights.png": (ORANGE, [
        ("3. Neural networks for solving PDEs/notebook_outputs/04_poisson_square.png", None,
         "A 2D Poisson PINN, its error and residual"),
        ("3. Neural networks for solving PDEs/notebook_outputs/extra_annulus_solution.png", (0.0, 0.07, 1.0, 1.0),
         "A mesh-free solution on an annulus"),
        ("3. Neural networks for solving PDEs/notebook_outputs/inverse_solution_family_and_valley.png", None,
         "An inverse problem: solution family and valley"),
    ]),
}


def load_on_white(path, crop):
    img = Image.open(path).convert("RGBA")
    white = Image.new("RGBA", img.size, "white")
    white.alpha_composite(img)
    img = white.convert("RGB")
    if crop:
        x0, y0, x1, y1 = crop
        img = img.crop((int(x0 * img.width), int(y0 * img.height), int(x1 * img.width), int(y1 * img.height)))
    # trim near-white margins
    diff = ImageChops.difference(img, Image.new("RGB", img.size, "white")).convert("L").point(lambda v: 255 if v > 12 else 0)
    box = diff.getbbox()
    return img.crop(box) if box else img


def strip(root, accent, items, out):
    card_w = (W - 2 * MARGIN - (len(items) - 1) * GAP) // len(items)
    canvas = Image.new("RGBA", (W, CARD_H + 2 * MARGIN), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    lm = latin_modern("lmsans10-regular.otf")
    font = ImageFont.truetype(lm, 25) if lm else ImageFont.load_default(25)
    for i, (rel, crop, caption) in enumerate(items):
        x = MARGIN + i * (card_w + GAP)
        y = MARGIN
        draw.rounded_rectangle((x, y, x + card_w, y + CARD_H), radius=16, fill="white", outline=BORDER, width=2)
        draw.rounded_rectangle((x, y, x + card_w, y + 7), radius=3, fill=accent)     # accent rule
        img = load_on_white(root / rel, crop)
        box_w, box_h = card_w - 2 * PAD, CARD_H - CAPTION_H - 2 * PAD - 6
        scale = min(box_w / img.width, box_h / img.height)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
        canvas.paste(img, (x + PAD + (box_w - img.width) // 2, y + PAD + 6 + (box_h - img.height) // 2))
        tw = draw.textlength(caption, font=font)
        draw.text((x + (card_w - tw) / 2, y + CARD_H - CAPTION_H + 10), caption, fill=NAVY, font=font)
    canvas.save(out, optimize=True)


def main(root, outdir):
    root, outdir = Path(root), Path(outdir)
    for name, (accent, items) in STRIPS.items():
        strip(root, accent, items, outdir / name)
        print("wrote", name)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ".")
