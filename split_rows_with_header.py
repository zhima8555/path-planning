from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class SplitConfig:
    rows: int = 4
    cols: int = 5
    dark_thresh: int = 60
    search_divisor: int = 60  # smaller => larger search window
    header_height: int = 90
    header_padding_y: int = 18
    font_size: int = 42
    border_band: int = 160
    border_pad: int = 30


def _dark_fraction_row(px, w: int, y: int, thresh: int) -> float:
    dark = 0
    for x in range(w):
        if px[x, y] < thresh:
            dark += 1
    return dark / w


def _dark_fraction_col(px, h: int, x: int, thresh: int) -> float:
    dark = 0
    for y in range(h):
        if px[x, y] < thresh:
            dark += 1
    return dark / h


def _argmin(values: list[float], lo: int, hi: int) -> int:
    best_i = lo
    best_v = values[lo]
    for i in range(lo + 1, hi + 1):
        v = values[i]
        if v < best_v:
            best_v = v
            best_i = i
    return best_i


def _argmax(values: list[float], lo: int, hi: int) -> int:
    best_i = lo
    best_v = values[lo]
    for i in range(lo + 1, hi + 1):
        v = values[i]
        if v > best_v:
            best_v = v
            best_i = i
    return best_i


def _find_font(preferred: Iterable[str], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in preferred:
        try:
            if Path(p).exists():
                return ImageFont.truetype(p, size=size)
        except Exception:
            pass
    # Fallback (may not render Chinese, but avoid crashing)
    return ImageFont.load_default()


def split_rows_with_header(
    in_path: Path,
    out_dir: Path | None = None,
    cfg: SplitConfig | None = None,
) -> list[Path]:
    cfg = cfg or SplitConfig()
    out_dir = out_dir or in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(in_path).convert("RGBA")
    w, h = img.size
    gray = img.convert("L")
    px = gray.load()

    # Compute darkness profiles
    row_dark = [_dark_fraction_row(px, w, y, cfg.dark_thresh) for y in range(h)]
    col_dark = [_dark_fraction_col(px, h, x, cfg.dark_thresh) for x in range(w)]

    # Find row gaps by minimizing darkness near expected cut positions
    search_y = max(10, h // cfg.search_divisor)
    expected_y = [h * i // cfg.rows for i in range(1, cfg.rows)]
    y_gaps = []
    for y0 in expected_y:
        lo = max(0, y0 - search_y)
        hi = min(h - 1, y0 + search_y)
        y_gaps.append(_argmin(row_dark, lo, hi))

    # Derive row panel bounds by snapping to strong border lines around each gap
    y_bounds: list[tuple[int, int]] = []

    # Row 1 top: first strong border line below the title area
    top0_lo, top0_hi = 0, min(h - 1, cfg.border_band)
    row1_top = _argmax(row_dark, top0_lo, top0_hi)

    tops = [row1_top]
    for gap in y_gaps:
        lo = max(0, gap)
        hi = min(h - 1, gap + cfg.border_band)
        tops.append(_argmax(row_dark, lo, hi))

    bottoms: list[int] = []
    for gap in y_gaps:
        lo = max(0, gap - cfg.border_band)
        hi = min(h - 1, gap)
        bottoms.append(_argmax(row_dark, lo, hi))

    # Row 4 bottom: last strong border line near image bottom
    bot_lo = max(0, h - 1 - cfg.border_band)
    bot_hi = h - 1
    bottoms.append(_argmax(row_dark, bot_lo, bot_hi))

    # Make (top, bottom) per row; ensure monotonic and non-empty
    for i in range(cfg.rows):
        t = max(0, tops[i] - cfg.border_pad)
        b = min(h - 1, bottoms[i] + cfg.border_pad)
        if b <= t:
            b = min(h - 1, t + 1)
        y_bounds.append((t, b))

    # Find column gaps by minimizing darkness near expected positions
    search_x = max(10, w // cfg.search_divisor)
    expected_x = [w * i // cfg.cols for i in range(1, cfg.cols)]
    x_gaps = []
    for x0 in expected_x:
        lo = max(0, x0 - search_x)
        hi = min(w - 1, x0 + search_x)
        x_gaps.append(_argmin(col_dark, lo, hi))

    # Compute column borders by looking for dark peaks on both sides of each gap.
    # This is more stable for centering text precisely above each panel.
    band = cfg.border_band
    left_outer = _argmax(col_dark, 0, min(w - 1, band))
    right_outer = _argmax(col_dark, max(0, w - 1 - band), w - 1)

    right_borders: list[int] = []
    left_borders: list[int] = []
    for xg in x_gaps:
        left_lo = max(0, xg - band)
        left_hi = xg
        right_lo = xg
        right_hi = min(w - 1, xg + band)
        right_borders.append(_argmax(col_dark, left_lo, left_hi))
        left_borders.append(_argmax(col_dark, right_lo, right_hi))

    # Borders per column (left_i, right_i)
    col_lefts: list[int] = [left_outer] + left_borders
    col_rights: list[int] = right_borders + [right_outer]

    x_centers: list[int] = [int((l + r) / 2) for l, r in zip(col_lefts, col_rights, strict=True)]

    titles = [
        "A*",
        "RRT*+APF",
        "PPO",
        "Dual-Att PPO",
        "混合算法",
    ]
    colors = [
        (31, 119, 180, 255),
        (214, 39, 40, 255),
        (44, 160, 44, 255),
        (255, 127, 14, 255),
        (148, 103, 189, 255),
    ]

    font = _find_font(
        [
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\arial.ttf",
        ],
        size=cfg.font_size,
    )

    out_paths: list[Path] = []
    for i, (top, bottom) in enumerate(y_bounds, start=1):
        # Crop row panels
        row_crop = img.crop((0, top, w, bottom + 1))

        # Create header
        header = Image.new("RGBA", (w, cfg.header_height), (255, 255, 255, 255))
        draw = ImageDraw.Draw(header)

        for t, xc, col in zip(titles, x_centers, colors, strict=True):
            bbox = draw.textbbox((0, 0), t, font=font, stroke_width=2)
            tw = bbox[2] - bbox[0]
            x = int(xc - tw / 2)
            y = cfg.header_padding_y
            draw.text(
                (x, y),
                t,
                font=font,
                fill=col,
                stroke_width=2,
                stroke_fill=(0, 0, 0, 255),
            )

        # Stack header + row
        combined = Image.new("RGBA", (w, cfg.header_height + row_crop.size[1]), (255, 255, 255, 255))
        combined.paste(header, (0, 0))
        combined.paste(row_crop, (0, cfg.header_height))

        out_path = out_dir / f"{in_path.stem}_row{i}_labeled.png"
        combined.save(out_path)
        out_paths.append(out_path)

    return out_paths


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Split a 4x5 composite figure into 4 row images and add algorithm header labels.")
    parser.add_argument("input", nargs="?", default="ablation/ablation_paths_separated.png")
    parser.add_argument("--out-dir", default="ablation")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    outs = split_rows_with_header(in_path=in_path, out_dir=out_dir)
    for p in outs:
        print(p.as_posix())


if __name__ == "__main__":
    main()
