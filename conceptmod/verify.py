"""Before/after verification grids: rows = prompts, columns = seeds, with
the frozen model's image directly above the trained model's image."""

from __future__ import annotations

import json
import os

from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

from conceptmod.dsl import describe_phrase

_FONT_DIR = "/usr/share/fonts/truetype/dejavu"
_CHROME_KEY = "conceptmod_chrome"


def _font(name: str, size: int) -> ImageFont.ImageFont:
    path = os.path.join(_FONT_DIR, name)
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    cur = words[0]
    for word in words[1:]:
        trial = f"{cur} {word}"
        if draw.textlength(trial, font=font) <= max_width:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    return lines


def label(img: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    font = _font("DejaVuSans.ttf", 12)
    bar_h = 20
    draw.rectangle([0, 0, img.width, bar_h], fill=(0, 0, 0))
    # ellipsize to the cell width
    shown = text
    while shown and draw.textlength(shown, font=font) > img.width - 8:
        shown = shown[:-1]
    if shown != text and len(shown) > 1:
        shown = shown[:-1] + "…"
    draw.text((4, 3), shown, fill=(255, 255, 255), font=font)
    return img


def frame_grid(grid: Image.Image, phrase: str, note: str) -> Image.Image:
    """Add a phrase header and intent footer around an existing grid."""
    pad_x, pad_y = 16, 12
    gap = 6
    width = grid.width
    inner = width - 2 * pad_x

    title_font = _font("DejaVuSans-Bold.ttf", 12)
    phrase_font = _font("DejaVuSansMono-Bold.ttf", 16)
    body_font = _font("DejaVuSans.ttf", 14)
    small_font = _font("DejaVuSans.ttf", 12)

    # measure with a throwaway draw
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    phrase_lines = _wrap(probe, phrase, phrase_font, inner) or [""]
    layout = (
        "Each pair: BEFORE (frozen model) on top, AFTER (trained) below. "
        "Columns are different seeds."
    )
    layout_lines = _wrap(probe, layout, small_font, inner)
    note_lines = _wrap(probe, note, body_font, inner) if note else []

    def block_h(lines, font, extra=0):
        if not lines:
            return 0
        bbox = font.getbbox("Ag")
        line_h = bbox[3] - bbox[1] + 4
        return extra + line_h * len(lines)

    header_h = (
        pad_y
        + block_h(["PHRASE"], title_font)
        + block_h(phrase_lines, phrase_font, extra=2)
        + block_h(layout_lines, small_font, extra=8)
        + pad_y
    )
    footer_h = (
        pad_y
        + block_h(["INTENT"], title_font)
        + block_h(note_lines, body_font, extra=2)
        + pad_y
    ) if note_lines else 0

    out = Image.new("RGB", (width, header_h + grid.height + footer_h), (10, 10, 10))
    draw = ImageDraw.Draw(out)

    y = pad_y
    draw.text((pad_x, y), "PHRASE", fill=(180, 160, 70), font=title_font)
    y += block_h(["PHRASE"], title_font)
    for line in phrase_lines:
        draw.text((pad_x, y), line, fill=(255, 220, 80), font=phrase_font)
        y += block_h([line], phrase_font)
    y += 8
    for line in layout_lines:
        draw.text((pad_x, y), line, fill=(160, 160, 160), font=small_font)
        y += block_h([line], small_font)

    out.paste(grid, (0, header_h))

    if note_lines:
        y = header_h + grid.height + pad_y
        draw.text((pad_x, y), "INTENT", fill=(180, 160, 70), font=title_font)
        y += block_h(["INTENT"], title_font)
        for line in note_lines:
            draw.text((pad_x, y), line, fill=(220, 220, 220), font=body_font)
            y += block_h([line], body_font)
    return out


def _save_framed(img: Image.Image, path: str) -> None:
    meta = PngImagePlugin.PngInfo()
    meta.add_text(_CHROME_KEY, "1")
    img.save(path, pnginfo=meta)


def annotate_grid_file(path: str, phrase: str, stage: str | None = None,
                       note: str | None = None) -> str:
    """Stamp phrase + intent onto an already-rendered grid. Idempotent."""
    img = Image.open(path)
    if img.info.get(_CHROME_KEY) == "1":
        return path
    img = img.convert("RGB")
    text = note or describe_phrase(phrase, stage=stage)
    framed = frame_grid(img, phrase, text)
    _save_framed(framed, path)
    return path


def annotate_output_dir(out_dir: str) -> list[str]:
    """Frame grid.png / grid_encoder.png in a run directory from its run.json."""
    run_path = os.path.join(out_dir, "run.json")
    with open(run_path) as f:
        run = json.load(f)
    phrase = run["phrase"]
    stage = run.get("stage")
    updated = []
    for name in ("grid.png", "grid_encoder.png"):
        path = os.path.join(out_dir, name)
        if os.path.isfile(path):
            annotate_grid_file(path, phrase, stage=stage)
            updated.append(path)
    return updated


def before_after_grid(backend, prompts: list[str], out_path: str,
                      seeds=(42, 1234), num_steps: int | None = None,
                      guidance: float | None = None, cell: int = 320,
                      phrase: str | None = None, stage: str | None = None,
                      note: str | None = None):
    """For each prompt render len(seeds) columns; each cell stacks
    before (top) / after (bottom)."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rows = []
    for prompt in prompts:
        cells = []
        shown = prompt if prompt else "(empty prompt)"
        for seed in seeds:
            before = backend.generate(prompt, seed, num_steps, guidance,
                                      frozen=True).resize((cell, cell))
            after = backend.generate(prompt, seed, num_steps, guidance,
                                     frozen=False).resize((cell, cell))
            pair = Image.new("RGB", (cell, cell * 2 + 2), (40, 40, 40))
            pair.paste(label(before, f"BEFORE  seed {seed}  {shown}"), (0, 0))
            pair.paste(label(after, f"AFTER   seed {seed}  {shown}"), (0, cell + 2))
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
    if phrase:
        grid = frame_grid(grid, phrase, note or describe_phrase(phrase, stage=stage))
        _save_framed(grid, out_path)
    else:
        grid.save(out_path)
    return out_path
