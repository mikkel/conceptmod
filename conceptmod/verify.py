"""Before/after verification grids: rows = prompts, columns = seeds, with
the frozen model's image directly above the trained model's image."""

from __future__ import annotations

import os

from PIL import Image, ImageDraw


def label(img: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 18], fill=(0, 0, 0))
    draw.text((4, 3), text[:90], fill=(255, 255, 255))
    return img


def before_after_grid(backend, prompts: list[str], out_path: str,
                      seeds=(42, 1234), num_steps: int | None = None,
                      guidance: float | None = None, cell: int = 320):
    """For each prompt render len(seeds) columns; each cell stacks
    before (top) / after (bottom)."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rows = []
    for prompt in prompts:
        cells = []
        for seed in seeds:
            before = backend.generate(prompt, seed, num_steps, guidance,
                                      frozen=True).resize((cell, cell))
            after = backend.generate(prompt, seed, num_steps, guidance,
                                     frozen=False).resize((cell, cell))
            pair = Image.new("RGB", (cell, cell * 2 + 2), (40, 40, 40))
            pair.paste(label(before, f"BEFORE  {prompt}"), (0, 0))
            pair.paste(label(after, f"AFTER   {prompt}"), (0, cell + 2))
            cells.append(pair)
        row = Image.new("RGB", (cell * len(cells) + 2 * (len(cells) - 1),
                                cell * 2 + 2), (40, 40, 40))
        for j, c in enumerate(cells):
            row.paste(c, (j * (cell + 2), 0))
        rows.append(row)
    grid = Image.new("RGB", (rows[0].width,
                             sum(r.height for r in rows) + 6 * (len(rows) - 1)),
                     (10, 10, 10))
    y = 0
    for r in rows:
        grid.paste(r, (0, y))
        y += r.height + 6
    grid.save(out_path)
    return out_path
