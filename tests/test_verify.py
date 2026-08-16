from pathlib import Path

from PIL import Image

from conceptmod.verify import (
    _CHROME_KEY,
    annotate_grid_file,
    annotate_output_dir,
    frame_grid,
)


def test_frame_grid_adds_header_and_footer():
    raw = Image.new("RGB", (200, 100), (30, 30, 30))
    framed = frame_grid(raw, "cat%dog:2", "Strip cat-features out of dogs.")
    assert framed.width == 200
    assert framed.height > 100
    # header/footer are near-black chrome, not the mid-grey cell
    assert framed.getpixel((10, 10))[0] < 20
    assert framed.getpixel((10, framed.height - 10))[0] < 20


def test_annotate_grid_file_is_idempotent(tmp_path: Path):
    path = tmp_path / "grid.png"
    Image.new("RGB", (80, 60), (80, 80, 80)).save(path)
    annotate_grid_file(str(path), "cat++", note="More cat.")
    first = Image.open(path)
    h1 = first.size[1]
    assert first.info.get(_CHROME_KEY) == "1"
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
    updated = annotate_output_dir(str(tmp_path))
    assert updated == [str(tmp_path / "grid.png")]
    img = Image.open(tmp_path / "grid.png")
    assert img.info.get(_CHROME_KEY) == "1"
    assert img.height > 60
