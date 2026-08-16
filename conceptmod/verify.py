"""Before/after verification grids: rows = prompt × seed, with the frozen
model's image on the left and the trained model's image on the right."""

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
_PAIR_LAYOUT_KEY = "conceptmod_pair_layout"
_PAD_X, _PAD_Y = 16, 12
_CELL = 320
_PAIR_GAP = 2
_COL_GAP = 2
_ROW_GAP = 6
_LABEL_H = 20
_BANNER_H = 22
_COLHEAD_H = 18

# First chrome layout (intent in the footer, pairs stacked top/bottom).
_LEGACY_LAYOUT = (
    "Each pair: BEFORE (frozen model) on top, AFTER (trained) below. "
    "Columns are different seeds."
)
_LAYOUT = (
    "Each pair: BEFORE (frozen model) on the left, AFTER (trained) on the right. "
    "Rows are one prompt × one seed."
)
_LAYOUT_CONTROL = (
    _LAYOUT
    + " CONTROL rows are unrelated prompts — the edit should leave them alone."
)

# Standard hold-out prompt used across the proofs.
_CONTROL_PROMPTS = {
    "a bowl of fruit on a table",
}


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


def _measure(phrase: str, note: str, width: int, *, intent_in_header: bool,
             layout: str = _LAYOUT) -> _Chrome:
    inner = width - 2 * _PAD_X
    title_font, phrase_font, body_font, small_font = _fonts()
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    phrase_lines = _wrap(probe, phrase, phrase_font, inner) or [""]
    layout_lines = _wrap(probe, layout, small_font, inner)
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
    chrome = _measure(phrase, note, width, intent_in_header=False,
                      layout=_LEGACY_LAYOUT)
    return chrome.header_h, chrome.footer_h


def is_control_prompt(prompt: str | None) -> bool:
    """True for the hold-out prompt used to check the rest of the model."""
    text = (prompt or "").strip().lower()
    return any(c in text for c in _CONTROL_PROMPTS)


def _layout_blurb(has_control: bool) -> str:
    return _LAYOUT_CONTROL if has_control else _LAYOUT


def label(img: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    font = _font("DejaVuSans.ttf", 12)
    draw.rectangle([0, 0, img.width, _LABEL_H], fill=(0, 0, 0))
    shown = text
    while shown and draw.textlength(shown, font=font) > img.width - 8:
        shown = shown[:-1]
    if shown != text and len(shown) > 1:
        shown = shown[:-1] + "…"
    draw.text((4, 3), shown, fill=(255, 255, 255), font=font)
    return img


def _ellipsize(draw, text: str, font, max_width: int) -> str:
    shown = text
    while shown and draw.textlength(shown, font=font) > max_width:
        shown = shown[:-1]
    if shown != text and len(shown) > 1:
        shown = shown[:-1] + "…"
    return shown


def _banner(width: int, prompt: str, *, control: bool) -> Image.Image:
    if control:
        bg, fg = (16, 48, 42), (170, 230, 190)
        text = (
            f"CONTROL  {prompt or '(empty prompt)'}  ·  "
            "unrelated prompt, the edit should leave this alone"
        )
    else:
        bg, fg = (16, 16, 16), (220, 220, 220)
        text = prompt if prompt else "(empty prompt)"
    img = Image.new("RGB", (width, _BANNER_H), bg)
    draw = ImageDraw.Draw(img)
    font = _font("DejaVuSans-Bold.ttf" if control else "DejaVuSans.ttf", 12)
    draw.text((6, 3), _ellipsize(draw, text, font, width - 12), fill=fg, font=font)
    return img


def _column_heads(width: int, cell: int, gap: int) -> Image.Image:
    img = Image.new("RGB", (width, _COLHEAD_H), (10, 10, 10))
    draw = ImageDraw.Draw(img)
    font = _font("DejaVuSans-Bold.ttf", 12)
    draw.text((6, 2), "BEFORE  (frozen)", fill=(180, 160, 70), font=font)
    draw.text((cell + gap + 6, 2), "AFTER  (trained)", fill=(180, 160, 70), font=font)
    return img


def _compose_leftright(
    rows: list[list[tuple[Image.Image, Image.Image]]],
    prompts: list[str | None],
    seeds: list[int],
    cell: int = _CELL,
) -> tuple[Image.Image, bool]:
    """Build a left/right grid. ``rows[i][j]`` is (before, after) for prompt i, seed j."""
    if not rows:
        raise ValueError("no rows to compose")
    width = cell * 2 + _PAIR_GAP
    has_control = any(is_control_prompt(p) for p in prompts)
    parts: list[Image.Image] = [_column_heads(width, cell, _PAIR_GAP)]
    for i, seed_pairs in enumerate(rows):
        prompt = prompts[i] if i < len(prompts) else None
        control = is_control_prompt(prompt)
        if prompt is not None:
            parts.append(_banner(width, prompt, control=control))
        for j, (before, after) in enumerate(seed_pairs):
            seed = seeds[j] if j < len(seeds) else j
            left = before.resize((cell, cell)).copy()
            right = after.resize((cell, cell)).copy()
            if prompt is not None:
                label(left, f"BEFORE  seed {seed}")
                label(right, f"AFTER   seed {seed}")
            pair = Image.new("RGB", (width, cell), (40, 40, 40))
            pair.paste(left, (0, 0))
            pair.paste(right, (cell + _PAIR_GAP, 0))
            parts.append(pair)
            if j < len(seed_pairs) - 1:
                parts.append(Image.new("RGB", (width, _PAIR_GAP), (40, 40, 40)))
        if i < len(rows) - 1:
            parts.append(Image.new("RGB", (width, _ROW_GAP), (10, 10, 10)))
    height = sum(p.height for p in parts)
    grid = Image.new("RGB", (width, height), (10, 10, 10))
    y = 0
    for part in parts:
        grid.paste(part, (0, y))
        y += part.height
    return grid, has_control


def _tb_geometry(width: int, height: int) -> tuple[int, int, int] | None:
    """Infer (cell, n_prompts, n_seeds) for a top/bottom pair grid."""
    for n_seeds in (2, 1, 3, 4):
        cell = (width + _COL_GAP) // n_seeds - _COL_GAP
        if cell < 16:
            continue
        if n_seeds * cell + _COL_GAP * (n_seeds - 1) != width:
            continue
        pair_h = cell * 2 + _PAIR_GAP
        if (height + _ROW_GAP) % (pair_h + _ROW_GAP) != 0:
            continue
        n_prompts = (height + _ROW_GAP) // (pair_h + _ROW_GAP)
        if n_prompts < 1:
            continue
        if n_prompts * pair_h + _ROW_GAP * (n_prompts - 1) == height:
            return cell, n_prompts, n_seeds
    return None


def _label_mse(bar: Image.Image, text: str) -> float:
    trial = Image.new("RGB", (bar.width, max(bar.height, _LABEL_H + 4)), (80, 80, 80))
    label(trial, text)
    crop = trial.crop((0, 0, bar.width, bar.height))
    pa, pb = list(bar.getdata()), list(crop.getdata())
    if not pa:
        return 1e9
    acc = 0.0
    for a, b in zip(pa, pb):
        acc += (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
    return acc / len(pa)


def _identify_row(
    before_cell: Image.Image, prompts: list[str], seeds: list[int],
) -> str | None:
    """Read the burned-in BEFORE label and match it to a prompt."""
    bar = before_cell.crop((0, 0, before_cell.width, min(_LABEL_H, before_cell.height)))
    best_p, best = None, 1e9
    for prompt in prompts:
        shown = prompt if prompt else "(empty prompt)"
        candidates = [f"BEFORE  seed {s}  {shown}" for s in seeds]
        candidates += [f"BEFORE  {shown}", f"BEFORE {shown}"]
        for text in candidates:
            score = _label_mse(bar, text)
            if score < best:
                best, best_p = score, prompt
    # lossless PNG of the same font/text is ~0; a wrong prompt is thousands
    return best_p if best < 50 else None


def _row_prompts_for_grid(
    n_prompts: int,
    prompts: list[str],
    identified: list[str | None],
) -> list[str | None]:
    """Assign verify prompts to extracted rows.

    Older ``grid_v*`` files sometimes have fewer rows than the final
    run.json (an extra probe was added later). Those older grids still
    end on the fruit control when it is present.
    """
    if n_prompts == len(prompts):
        return list(prompts)
    if all(p is not None for p in identified):
        return identified
    control = next((p for p in prompts if is_control_prompt(p)), None)
    probes = [p for p in prompts if not is_control_prompt(p)]
    if control is not None and n_prompts >= 1 and n_prompts - 1 <= len(probes):
        return probes[: n_prompts - 1] + [control]
    return identified


def relayout_topbottom_to_leftright(
    img: Image.Image,
    prompts: list[str] | None = None,
    seeds: list[int] | tuple[int, ...] | None = None,
) -> tuple[Image.Image, bool]:
    """Slice a top/bottom pair grid into left/right. No-op if geometry is unknown."""
    prompts = list(prompts or [])
    seeds = list(seeds or (42, 1234))
    geo = _tb_geometry(img.width, img.height)
    if geo is None:
        return img, any(is_control_prompt(p) for p in prompts)
    cell, n_prompts, n_seeds = geo
    pair_h = cell * 2 + _PAIR_GAP
    extracted: list[list[tuple[Image.Image, Image.Image]]] = []
    row_prompts: list[str | None] = []
    for i in range(n_prompts):
        y = i * (pair_h + _ROW_GAP)
        seed_pairs = []
        first_before = None
        for j in range(n_seeds):
            x = j * (cell + _COL_GAP)
            before = img.crop((x, y, x + cell, y + cell))
            after = img.crop((x, y + cell + _PAIR_GAP, x + cell, y + pair_h))
            seed_pairs.append((before, after))
            if j == 0:
                first_before = before
        extracted.append(seed_pairs)
        if first_before is not None and prompts:
            row_prompts.append(_identify_row(first_before, prompts, seeds))
        else:
            row_prompts.append(None)
    row_prompts = _row_prompts_for_grid(n_prompts, prompts, row_prompts)
    return _compose_leftright(extracted, row_prompts, seeds[:n_seeds], cell=cell)


def frame_grid(grid: Image.Image, phrase: str, note: str,
               layout: str | None = None) -> Image.Image:
    """Stamp a glanceable ATTEMPTING header (plus the phrase) above a grid."""
    width = grid.width
    blurb = layout if layout is not None else _LAYOUT
    chrome = _measure(phrase, note, width, intent_in_header=True, layout=blurb)
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


def _save_framed(img: Image.Image, path: str, phrase: str, note: str,
                 pair_layout: str = "lr", layout: str | None = None) -> None:
    blurb = layout if layout is not None else _LAYOUT
    chrome = _measure(phrase, note, img.width, intent_in_header=True, layout=blurb)
    meta = PngImagePlugin.PngInfo()
    meta.add_text(_CHROME_KEY, "1")
    meta.add_text(_HEADER_H_KEY, str(chrome.header_h))
    meta.add_text(_FOOTER_H_KEY, str(chrome.footer_h))
    meta.add_text(_PAIR_LAYOUT_KEY, pair_layout)
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
                       note: str | None = None,
                       prompts: list[str] | None = None,
                       seeds: list[int] | tuple[int, ...] | None = None) -> str:
    """Stamp phrase + attempt onto an already-rendered grid. Idempotent.
    Also converts leftover top/bottom pair grids to left/right."""
    src = Image.open(path)
    info = dict(src.info)
    img = src.convert("RGB")
    src.close()
    text = note or describe_phrase(phrase, stage=stage)
    body = _unframe(img, info, phrase, text)
    pair_layout = info.get(_PAIR_LAYOUT_KEY)
    has_control = any(is_control_prompt(p) for p in (prompts or []))
    if pair_layout != "lr":
        body, found = relayout_topbottom_to_leftright(
            body, prompts=prompts, seeds=seeds)
        has_control = has_control or found
    blurb = _layout_blurb(has_control)
    framed = frame_grid(body, phrase, text, layout=blurb)
    _save_framed(framed, path, phrase, text, pair_layout="lr", layout=blurb)
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
    prompts = run.get("verify_prompt") or []
    seeds = run.get("verify_seeds") or [42, 1234]
    updated = []
    for name in _grid_pngs(out_dir):
        path = os.path.join(out_dir, name)
        annotate_grid_file(path, phrase, stage=stage, prompts=prompts, seeds=seeds)
        updated.append(path)
    return updated


def before_after_grid(backend, prompts: list[str], out_path: str,
                      seeds=(42, 1234), num_steps: int | None = None,
                      guidance: float | None = None, cell: int = _CELL,
                      phrase: str | None = None, stage: str | None = None,
                      note: str | None = None):
    """For each prompt render len(seeds) rows; each row is BEFORE | AFTER."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rows: list[list[tuple[Image.Image, Image.Image]]] = []
    for prompt in prompts:
        seed_pairs = []
        for seed in seeds:
            before = backend.generate(prompt, seed, num_steps, guidance,
                                      frozen=True).resize((cell, cell))
            after = backend.generate(prompt, seed, num_steps, guidance,
                                     frozen=False).resize((cell, cell))
            seed_pairs.append((before, after))
        rows.append(seed_pairs)
    grid, has_control = _compose_leftright(
        rows, list(prompts), list(seeds), cell=cell)
    if phrase:
        text = note or describe_phrase(phrase, stage=stage)
        blurb = _layout_blurb(has_control)
        grid = frame_grid(grid, phrase, text, layout=blurb)
        _save_framed(grid, out_path, phrase, text, pair_layout="lr", layout=blurb)
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
