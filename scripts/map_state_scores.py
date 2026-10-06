#!/usr/bin/env python3
"""Combine one state's coordinate-enriched prediction parts and map each score.

Reads every ``predictions_USA_XX_part_NNNNN_with_coordinates.csv`` in one state
folder, stacks them into a single table, turns each patch center into its
512 x 512 pixel footprint (0.6 m pixels, so 307.2 m squares in EPSG:5070),
averages the scores of overlapping patches on a regular grid, and writes:

- ``combined_USA_XX.csv``      all rows from every part, in part order
- ``patch_scores_USA_XX.gpkg`` one square polygon per patch, every column kept
- ``mean_scores_USA_XX.tif``   one band per mapped score (mean of the patches
                               covering each cell) plus a ``patch_count`` band
- ``map_USA_XX_<score>.png``   one static map per mapped score, from that grid
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PART_NAME = re.compile(r"^predictions_USA_([A-Z]{2})_part_(\d{5})_with_coordinates\.csv$")
CRS = "EPSG:5070"
PATCH_PIXELS = 512
PIXEL_SIZE_M = 0.6
PATCH_SIZE_M = PATCH_PIXELS * PIXEL_SIZE_M
DEFAULT_CELL_SIZE_M = PATCH_SIZE_M / 2

# Score column -> (map title, colour-scale minimum, maximum).
SCORES = {
    "score_ev": ("Structured / unstructured score", 1.0, 5.0),
    "veg_ev": ("Vegetation distribution score", 1.0, 5.0),
    "sports_field_prob": ("Sports field probability", 0.0, 1.0),
    "multipurpose_open_area_prob": ("Multipurpose open area probability", 0.0, 1.0),
    "children_s_playground_prob": ("Children's playground probability", 0.0, 1.0),
    "water_feature_prob": ("Water feature probability", 0.0, 1.0),
    "walking_paths_prob": ("Walking paths probability", 0.0, 1.0),
    "built_structures_prob": ("Built structures probability", 0.0, 1.0),
    "parking_lots_prob": ("Parking lots probability", 0.0, 1.0),
    "shade_confidence": ("Shade class confidence", 0.0, 1.0),
}
REQUIRED_COLUMNS = ("image_relative_path", "state_code", "center_x", "center_y", *SCORES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--predictions-dir", required=True,
        help="One state's folder containing predictions_USA_XX_part_*_with_coordinates.csv",
    )
    parser.add_argument(
        "--output-dir",
        help="Destination for the combined CSV, GeoPackage, GeoTIFF and maps "
        "(default: <predictions-dir>/maps)",
    )
    parser.add_argument(
        "--scores", nargs="+", choices=sorted(SCORES), default=list(SCORES),
        help="Score columns to map (default: all)",
    )
    parser.add_argument(
        "--cell-size", type=float, default=DEFAULT_CELL_SIZE_M,
        help="Averaging grid cell size in metres; patches snap to this grid "
        f"(default: {DEFAULT_CELL_SIZE_M:g}, half a patch)",
    )
    parser.add_argument("--dpi", type=int, default=200, help="PNG resolution (default: 200)")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace outputs from an earlier run",
    )
    return parser


def find_parts(predictions_dir: Path) -> tuple[str, list[Path]]:
    if not predictions_dir.is_dir():
        raise FileNotFoundError(f"Missing predictions directory: {predictions_dir}")
    matches = sorted(
        (int(m.group(2)), m.group(1), path)
        for path in predictions_dir.iterdir()
        if path.is_file() and (m := PART_NAME.fullmatch(path.name))
    )
    if not matches:
        raise FileNotFoundError(
            f"No *_with_coordinates.csv prediction parts in {predictions_dir}; "
            "run scripts/append_patch_coordinates.py first"
        )
    states = {state for _, state, _ in matches}
    if len(states) != 1:
        raise ValueError("--predictions-dir must contain parts for exactly one state")
    return states.pop(), [path for _, _, path in matches]


def combine_parts(parts: list[Path], state: str) -> pd.DataFrame:
    """Stack all parts (the rbind step), checking they share one schema."""
    frames = []
    columns: list[str] | None = None
    for part in parts:
        frame = pd.read_csv(
            part, encoding="utf-8-sig", dtype={"park_code": str, "state_code": str},
            keep_default_na=False, na_values=[""],
        )
        missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"Missing columns in {part}: {', '.join(missing)}")
        if columns is None:
            columns = list(frame.columns)
        elif list(frame.columns) != columns:
            raise ValueError(f"Columns in {part} differ from {parts[0].name}")
        frame.insert(0, "source_part", part.name)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if (combined["state_code"] != state).any():
        raise ValueError(f"Rows with a state_code other than {state}")
    duplicated = combined["image_relative_path"].str.casefold().duplicated()
    if duplicated.any():
        first = combined.loc[duplicated, "image_relative_path"].iloc[0]
        raise ValueError(f"Duplicate prediction across parts: {first}")
    for column in ("center_x", "center_y", *SCORES):
        combined[column] = pd.to_numeric(combined[column], errors="raise")
    return combined


def patch_squares(frame: pd.DataFrame):
    import geopandas as gpd
    import shapely

    half = PATCH_SIZE_M / 2
    x = frame["center_x"].to_numpy()
    y = frame["center_y"].to_numpy()
    geometry = shapely.box(x - half, y - half, x + half, y + half)
    return gpd.GeoDataFrame(frame, geometry=geometry, crs=CRS)


class PatchGrid:
    """Regular grid over the patches; each patch covers a fixed block of cells.

    A patch's footprint snaps to the nearest cell edges, so with the default
    half-patch cell it covers exactly 2 x 2 cells, shifted by at most half a cell.
    """

    def __init__(self, frame: pd.DataFrame, cell_size: float) -> None:
        if cell_size <= 0 or cell_size > PATCH_SIZE_M:
            raise ValueError(f"--cell-size must be in (0, {PATCH_SIZE_M:g}] metres")
        half = PATCH_SIZE_M / 2
        x = frame["center_x"].to_numpy()
        y = frame["center_y"].to_numpy()
        self.cell_size = cell_size
        self.cells_per_patch = max(1, round(PATCH_SIZE_M / cell_size))
        # Snap each footprint's top-left corner to multiples of the cell size.
        cols = np.rint((x - half) / cell_size).astype(np.int64)
        rows = np.rint(-(y + half) / cell_size).astype(np.int64)
        self.left = float(cols.min() * cell_size)
        self.top = float(-rows.min() * cell_size)
        self.cols = cols - cols.min()
        self.rows = rows - rows.min()
        self.shape = (
            int(self.rows.max()) + self.cells_per_patch,
            int(self.cols.max()) + self.cells_per_patch,
        )

    @property
    def extent(self) -> tuple[float, float, float, float]:
        n_rows, n_cols = self.shape
        return (
            self.left, self.left + n_cols * self.cell_size,
            self.top - n_rows * self.cell_size, self.top,
        )

    def _accumulate(self, keep: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
        n_cols = self.shape[1]
        offsets = np.arange(self.cells_per_patch)
        cells = (
            (self.rows[keep, None, None] + offsets[None, :, None]) * n_cols
            + self.cols[keep, None, None] + offsets[None, None, :]
        )
        repeated = None
        if weights is not None:
            repeated = np.broadcast_to(weights[keep, None, None], cells.shape).ravel()
        total = np.bincount(cells.ravel(), weights=repeated, minlength=n_cols * self.shape[0])
        return total.reshape(self.shape)

    def count(self) -> np.ndarray:
        return self._accumulate(np.ones(len(self.rows), dtype=bool), None)

    def mean(self, values: np.ndarray) -> np.ndarray:
        """Mean of every patch covering each cell; NaN where no patch has a value."""
        keep = ~np.isnan(values)
        total = self._accumulate(keep, values)
        count = self._accumulate(keep, None)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (total / count).astype(np.float32)


def write_geotiff(path: Path, grid: PatchGrid, bands: dict[str, np.ndarray]) -> None:
    import rasterio
    from rasterio.transform import from_origin

    profile = {
        "driver": "GTiff", "height": grid.shape[0], "width": grid.shape[1],
        "count": len(bands), "dtype": "float32", "crs": CRS, "nodata": np.nan,
        "transform": from_origin(grid.left, grid.top, grid.cell_size, grid.cell_size),
        "compress": "deflate", "tiled": True, "blockxsize": 256, "blockysize": 256,
    }
    if grid.shape[0] < 256 or grid.shape[1] < 256:
        profile.update(tiled=False)
        del profile["blockxsize"], profile["blockysize"]
    with rasterio.open(path, "w", **profile) as dataset:
        for index, (name, band) in enumerate(bands.items(), start=1):
            dataset.write(band.astype(np.float32), index)
            dataset.set_band_description(index, name)


def draw_map(
    grid: PatchGrid, values: np.ndarray, patches: int, score: str, state: str,
    path: Path, dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    title, vmin, vmax = SCORES[score]
    # One hue, light to dark; trimmed so the lowest values stay visible on white.
    cmap = LinearSegmentedColormap.from_list(
        "greens_trimmed", plt.get_cmap("Greens")(np.linspace(0.2, 1.0, 256)),
    )
    cmap.set_bad((1, 1, 1, 0))
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8, 9), dpi=dpi)
    pad = 5_000
    left, right, bottom, top = grid.extent
    ax.imshow(
        np.ma.masked_invalid(values), cmap=cmap, norm=norm, extent=grid.extent,
        origin="upper", interpolation="nearest",
    )
    ax.set_xlim(left - pad, right + pad)
    ax.set_ylim(bottom - pad, top + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(f"USA_{state} — {title}", loc="left", fontsize=13, color="#222222")
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
        orientation="horizontal", fraction=0.04, pad=0.03,
    )
    colorbar.set_label(f"{score} (mean of overlapping patches)", color="#555555")
    colorbar.outline.set_visible(False)
    fig.text(
        0.01, 0.01,
        f"{patches:,} patches · {PATCH_SIZE_M:g} m squares averaged on a "
        f"{grid.cell_size:g} m grid · {CRS}",
        fontsize=8, color="#777777",
    )
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def map_state(
    predictions_dir: Path,
    output_dir: Path,
    scores: list[str] | None = None,
    dpi: int = 200,
    overwrite: bool = False,
    cell_size: float = DEFAULT_CELL_SIZE_M,
) -> dict[str, int | str]:
    scores = scores or list(SCORES)
    state, parts = find_parts(predictions_dir)
    combined_csv = output_dir / f"combined_USA_{state}.csv"
    gpkg = output_dir / f"patch_scores_USA_{state}.gpkg"
    tif = output_dir / f"mean_scores_USA_{state}.tif"
    pngs = [output_dir / f"map_USA_{state}_{score}.png" for score in scores]
    if not overwrite:
        existing = [path for path in (combined_csv, gpkg, tif, *pngs) if path.exists()]
        if existing:
            raise FileExistsError(f"Output already exists: {existing[0]}; use --overwrite")

    combined = combine_parts(parts, state)
    located = combined.dropna(subset=["center_x", "center_y"])
    missing = len(combined) - len(located)
    if located.empty:
        raise ValueError(f"No rows with coordinates for USA_{state}")
    grid = PatchGrid(located, cell_size)

    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(combined_csv, index=False)
    if gpkg.exists():
        gpkg.unlink()
    patch_squares(located).to_file(gpkg, layer=f"patches_USA_{state}", driver="GPKG")
    count = grid.count()
    means = {score: grid.mean(located[score].to_numpy(dtype=float)) for score in scores}
    write_geotiff(tif, grid, {**means, "patch_count": np.where(count > 0, count, np.nan)})
    for score, png in zip(scores, pngs):
        draw_map(grid, means[score], len(located), score, state, png, dpi)
    return {
        "state": state, "parts": len(parts), "rows": len(combined),
        "mapped": len(located), "missing_coordinates": missing,
        "max_overlap": int(count.max()),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    predictions_dir = Path(args.predictions_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve() if args.output_dir
        else predictions_dir / "maps"
    )
    try:
        summary = map_state(
            predictions_dir, output_dir, args.scores, args.dpi, args.overwrite,
            args.cell_size,
        )
    except (OSError, ValueError) as exc:
        print(f"Mapping failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"USA_{summary['state']}: combined {summary['rows']} rows from "
        f"{summary['parts']} parts; mapped {summary['mapped']} patches "
        f"({len(args.scores)} maps, up to {summary['max_overlap']} patches "
        f"averaged per cell) to {output_dir}"
    )
    if summary["missing_coordinates"]:
        print(
            f"WARNING: {summary['missing_coordinates']} rows have no coordinates and "
            "were left out of the GeoPackage and maps (kept in the combined CSV)",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
