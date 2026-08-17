from pathlib import Path

from PIL import Image

from conceptmod.verify import (
    _CHROME_KEY,
    _PAIR_LAYOUT_KEY,
    annotate_grid_file,
    annotate_output_dir,
    frame_grid,
    is_control_prompt,
    relayout_topbottom_to_leftright,
)


def test_frame_grid_puts_attempt_in_the_header():
    raw = Image.new("RGB", (400, 100), (30, 30, 30))
    framed = frame_grid(raw, "cat%dog:2", "Strip cat-features out of dogs.")
    assert framed.width == 400
    assert framed.height > 100
    # header is near-black chrome; the cell sits at the bottom (no footer)
    assert framed.getpixel((10, 10))[0] < 20
    assert framed.getpixel((10, framed.height - 10)) == (30, 30, 30)
    # yellow phrase ink and near-white attempt ink are actually painted
    pixels = list(framed.getdata())
    assert any(p[0] > 200 and p[1] > 180 and p[2] < 120 for p in pixels)
    assert any(p[0] > 200 and p[1] > 200 and p[2] > 200 for p in pixels)


def test_frame_grid_header_grows_with_attempt():
    raw = Image.new("RGB", (400, 80), (30, 30, 30))
    short = frame_grid(raw, "cat++", "More cat.")
    long = frame_grid(raw, "cat++", "More cat. " * 20)
    assert long.height > short.height


def test_annotate_grid_file_is_idempotent(tmp_path: Path):
    path = tmp_path / "grid.png"
    Image.new("RGB", (80, 60), (80, 80, 80)).save(path)
    annotate_grid_file(str(path), "cat++", note="More cat.")
    first = Image.open(path)
    h1 = first.size[1]
    assert first.info.get(_CHROME_KEY) == "1"
    assert first.info.get("conceptmod_note") == "More cat."
    first.close()
    annotate_grid_file(str(path), "cat++", note="More cat.")
    second = Image.open(path)
    assert second.size[1] == h1
    second.close()


def test_annotate_output_dir_reads_run_json(tmp_path: Path):
    (tmp_path / "run.json").write_text(
        '{"phrase": "monochrome--", "stage": "model"}'
    )
    Image.new("RGB", (80, 60), (80, 80, 80)).save(tmp_path / "grid.png")
    Image.new("RGB", (80, 60), (80, 80, 80)).save(tmp_path / "grid_v1.png")
    updated = annotate_output_dir(str(tmp_path))
    assert updated == [str(tmp_path / "grid.png"), str(tmp_path / "grid_v1.png")]
    img = Image.open(tmp_path / "grid.png")
    assert img.info.get(_CHROME_KEY) == "1"
    assert img.height > 60
    assert "Erase" in img.info.get("conceptmod_note", "")


def test_fruit_is_the_control_prompt():
    assert is_control_prompt("a bowl of fruit on a table")
    assert is_control_prompt("A Bowl of Fruit on a Table")
    assert not is_control_prompt("a photo of a cat")
    assert not is_control_prompt("")


def _tb_grid(cell=40):
    """One prompt, two seeds, before on top / after below."""
    colors = {
        "b0": (220, 30, 30),
        "b1": (30, 220, 30),
        "a0": (30, 30, 220),
        "a1": (220, 220, 30),
    }
    w = cell * 2 + 2
    h = cell * 2 + 2
    img = Image.new("RGB", (w, h), (10, 10, 10))
    img.paste(Image.new("RGB", (cell, cell), colors["b0"]), (0, 0))
    img.paste(Image.new("RGB", (cell, cell), colors["b1"]), (cell + 2, 0))
    img.paste(Image.new("RGB", (cell, cell), colors["a0"]), (0, cell + 2))
    img.paste(Image.new("RGB", (cell, cell), colors["a1"]), (cell + 2, cell + 2))
    return img, colors


def test_relayout_puts_before_on_the_left():
    raw, colors = _tb_grid(cell=40)
    out, has_control = relayout_topbottom_to_leftright(
        raw, prompts=["a photo of a cat"], seeds=[42, 1234],
    )
    assert not has_control
    assert out.width == 40 * 2 + 2
    # skip column head (18) + banner (22); sample mid-cell below the label bar
    y = 18 + 22 + 25
    assert out.getpixel((10, y))[:3] == colors["b0"]
    assert out.getpixel((50, y))[:3] == colors["a0"]
    y2 = 18 + 22 + 40 + 2 + 25
    assert out.getpixel((10, y2))[:3] == colors["b1"]
    assert out.getpixel((50, y2))[:3] == colors["a1"]


def test_relayout_older_grid_still_tags_fruit():
    """Fewer rows than run.json: last row is still the fruit control."""
    cell = 40
    # two stacked prompt pairs (probe + fruit)
    pair_h = cell * 2 + 2
    raw = Image.new("RGB", (cell * 2 + 2, pair_h * 2 + 6), (10, 10, 10))
    raw.paste(Image.new("RGB", (cell, cell), (200, 0, 0)), (0, 0))
    raw.paste(Image.new("RGB", (cell, cell), (0, 200, 0)), (0, pair_h + 6))
    out, has_control = relayout_topbottom_to_leftright(
        raw,
        prompts=[
            "a photo of a cat",
            "a cat sitting on a windowsill",  # added in a later round
            "a bowl of fruit on a table",
        ],
        seeds=[42, 1234],
    )
    assert has_control
    # second prompt banner (after colhead + banner + 2 seed rows + gap) is teal
    y_first_banner = 18 + 10
    y_second_banner = 18 + 22 + (cell + 2 + cell) + 6 + 10
    first = out.getpixel((8, y_first_banner))
    second = out.getpixel((8, y_second_banner))
    assert first[1] <= 30  # normal banner is near-black
    assert second[1] > 30 and second[1] > second[0]


def test_relayout_marks_fruit_as_control():
    raw, _ = _tb_grid(cell=40)
    out, has_control = relayout_topbottom_to_leftright(
        raw, prompts=["a bowl of fruit on a table"], seeds=[42, 1234],
    )
    assert has_control
    # control banner is teal, not near-black
    banner = out.getpixel((8, 18 + 10))
    assert banner[1] > 30 and banner[1] > banner[0]


def test_annotate_converts_topbottom_and_is_stable(tmp_path: Path):
    raw, colors = _tb_grid(cell=40)
    path = tmp_path / "grid.png"
    raw.save(path)
    (tmp_path / "run.json").write_text(
        '{"phrase": "cat=dog", "stage": "model",'
        ' "verify_prompt": ["a bowl of fruit on a table"],'
        ' "verify_seeds": [42, 1234]}'
    )
    annotate_output_dir(str(tmp_path))
    first = Image.open(path)
    assert first.info.get(_PAIR_LAYOUT_KEY) == "lr"
    assert first.info.get(_CHROME_KEY) == "1"
    h1 = first.size[1]
    first.close()
    annotate_output_dir(str(tmp_path))
    second = Image.open(path)
    assert second.size[1] == h1
    second.close()
