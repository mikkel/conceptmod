"""Before/after verification grids: rows = prompts, columns = seeds, with
the frozen model's image directly above the trained model's image."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

from conceptmod.dsl import describe_phrase

_FONT_DIR = "/usr/share/fonts/truetype/dejavu"
_CHROME_KEY = "conceptmod_chrome"
_HEADER_H_KEY = "conceptmod_header_h"
_FOOTER_H_KEY = "conceptmod_footer_h"
_LAYOUT = (
    "Each pair: BEFORE (frozen model) on top, AFTER (trained) below. "
    "Columns are different seeds."
)
_PAD_X, _PAD_Y = 16, 12


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


def _block_h(lines, font, extra=0) -> int:
    if not lines:
        return 0
    bbox = font.getbbox("Ag")
    line_h = bbox[3] - bbox[1] + 4
    return extra + line_h * len(lines)


def _fonts():
    return (
        _font("DejaVuSans-Bold.ttf", 12),
        _font("DejaVuSansMono-Bold.ttf", 16),
        _font("DejaVuSans.ttf", 14),
        _font("DejaVuSans.ttf", 12),
    )


@dataclass(frozen=True)
class _Chrome:
    header_h: int
    footer_h: int
    phrase_lines: list[str]
    note_lines: list[str]
    layout_lines: list[str]


def _measure(phrase: str, note: str, width: int, *, intent_in_header: bool) -> _Chrome:
    inner = width - 2 * _PAD_X
    title_font, phrase_font, body_font, small_font = _fonts()
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    phrase_lines = _wrap(probe, phrase, phrase_font, inner) or [""]
    layout_lines = _wrap(probe, _LAYOUT, small_font, inner)
    note_lines = _wrap(probe, note, body_font, inner) if note else []

    phrase_block = (
        _block_h(["PHRASE"], title_font)
        + _block_h(phrase_lines, phrase_font, extra=2)
        + _block_h(layout_lines, small_font, extra=8)
    )
    note_block = (
        (_block_h(["ATTEMPTING"], title_font)
         + _block_h(note_lines, body_font, extra=2))
        if note_lines else 0
    )

    if intent_in_header:
        header_h = _PAD_Y + note_block
        if note_block:
            header_h += 8
        header_h += phrase_block + _PAD_Y
        footer_h = 0
    else:
        header_h = _PAD_Y + phrase_block + _PAD_Y
        footer_h = (_PAD_Y + note_block + _PAD_Y) if note_block else 0
    return _Chrome(header_h, footer_h, phrase_lines, note_lines, layout_lines)


def _legacy_chrome_heights(phrase: str, note: str, width: int) -> tuple[int, int]:
    """Header/footer written by the first chrome layout (intent in the footer)."""
    chrome = _measure(phrase, note, width, intent_in_header=False)
    return chrome.header_h, chrome.footer_h


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
    """Stamp a glanceable ATTEMPTING header (plus the phrase) above a grid."""
    width = grid.width
    chrome = _measure(phrase, note, width, intent_in_header=True)
    title_font, phrase_font, body_font, small_font = _fonts()

    out = Image.new("RGB", (width, chrome.header_h + grid.height), (10, 10, 10))
    draw = ImageDraw.Draw(out)

    y = _PAD_Y
    if chrome.note_lines:
        draw.text((_PAD_X, y), "ATTEMPTING", fill=(180, 160, 70), font=title_font)
        y += _block_h(["ATTEMPTING"], title_font)
        for line in chrome.note_lines:
            draw.text((_PAD_X, y), line, fill=(220, 220, 220), font=body_font)
            y += _block_h([line], body_font)
        y += 8
    draw.text((_PAD_X, y), "PHRASE", fill=(180, 160, 70), font=title_font)
    y += _block_h(["PHRASE"], title_font)
    for line in chrome.phrase_lines:
        draw.text((_PAD_X, y), line, fill=(255, 220, 80), font=phrase_font)
        y += _block_h([line], phrase_font)
    y += 8
    for line in chrome.layout_lines:
        draw.text((_PAD_X, y), line, fill=(160, 160, 160), font=small_font)
        y += _block_h([line], small_font)

    out.paste(grid, (0, chrome.header_h))
    return out


def _save_framed(img: Image.Image, path: str, phrase: str, note: str) -> None:
    chrome = _measure(phrase, note, img.width, intent_in_header=True)
    meta = PngImagePlugin.PngInfo()
    meta.add_text(_CHROME_KEY, "1")
    meta.add_text(_HEADER_H_KEY, str(chrome.header_h))
    meta.add_text(_FOOTER_H_KEY, str(chrome.footer_h))
    meta.add_text("conceptmod_phrase", phrase)
    meta.add_text("conceptmod_note", note)
    img.save(path, pnginfo=meta)


def _unframe(img: Image.Image, info: dict, phrase: str, note: str) -> Image.Image:
    """Strip previously stamped chrome so a grid can be restamped."""
    if info.get(_CHROME_KEY) != "1":
        return img
    if _HEADER_H_KEY in info and _FOOTER_H_KEY in info:
        header_h = int(info[_HEADER_H_KEY])
        footer_h = int(info[_FOOTER_H_KEY])
    else:
        header_h, footer_h = _legacy_chrome_heights(phrase, note, img.width)
    if header_h < 0 or footer_h < 0:
        return img
    if header_h + footer_h >= img.height - 50:
        return img
    return img.crop((0, header_h, img.width, img.height - footer_h))


def annotate_grid_file(path: str, phrase: str, stage: str | None = None,
                       note: str | None = None) -> str:
    """Stamp phrase + attempt onto an already-rendered grid. Idempotent."""
    src = Image.open(path)
    info = dict(src.info)
    img = src.convert("RGB")
    src.close()
    text = note or describe_phrase(phrase, stage=stage)
    body = _unframe(img, info, phrase, text)
    framed = frame_grid(body, phrase, text)
    _save_framed(framed, path, phrase, text)
    return path


def _grid_pngs(out_dir: str) -> list[str]:
    names = []
    for name in sorted(os.listdir(out_dir)):
        if name == "grid.png" or (name.startswith("grid") and name.endswith(".png")):
            names.append(name)
    return names


def annotate_output_dir(out_dir: str) -> list[str]:
    """Frame every grid*.png in a run directory from its run.json."""
    run_path = os.path.join(out_dir, "run.json")
    with open(run_path) as f:
        run = json.load(f)
    phrase = run["phrase"]
    stage = run.get("stage")
    updated = []
    for name in _grid_pngs(out_dir):
        path = os.path.join(out_dir, name)
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
        text = note or describe_phrase(phrase, stage=stage)
        grid = frame_grid(grid, phrase, text)
        _save_framed(grid, out_path, phrase, text)
    else:
        grid.save(out_path)
    return out_path


def annotate_outputs_root(root: str = "outputs") -> list[str]:
    """Stamp every run directory under ``root`` that has a run.json."""
    updated = []
    if os.path.isfile(os.path.join(root, "run.json")):
        return annotate_output_dir(root)
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "run.json")):
            updated.extend(annotate_output_dir(path))
    return updated


if __name__ == "__main__":
    import sys
    targets = sys.argv[1:] or ["outputs"]
    for target in targets:
        for path in annotate_outputs_root(target):
            print(path)
