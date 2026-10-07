#!/usr/bin/env python3
"""Combine one state's coordinate-enriched prediction parts and map each score.

Reads every ``predictions_USA_XX_part_NNNNN_with_coordinates.csv`` in one state
folder, stacks them into a single table, turns each patch center into its
512 x 512 pixel footprint (0.6 m pixels, so 307.2 m squares in EPSG:5070),
optionally keeps only the parts of each footprint inside park polygons
(``--parks``), averages the scores of overlapping patches on a regular grid,
and writes:

- ``combined_USA_XX.csv``      all rows from every part, in part order
                               (plus ``park_fraction`` with ``--parks``)
- ``patch_scores_USA_XX.gpkg`` one polygon per patch, every column kept; with
                               ``--parks`` clipped to the parks, and patches
                               outside every park left out
- ``mean_scores_USA_XX.tif``   one band per mapped score (mean of the patches
                               covering each cell) plus a ``patch_count`` band;
                               with ``--parks`` only cells inside parks
- ``map_USA_XX_<score>.png``   one static map per mapped score, from that grid
"""

from __future__ import annotations

import argparse
import math
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
DEFAULT_CELL_SIZE_M = PATCH_SIZE_M / 8
MAP_PIXELS = 1600

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
        "--parks",
        help="Park polygon file (e.g. ParkServe_Parks.shp); only the parts of each "
        "patch inside a park are kept. Any CRS; reprojected to EPSG:5070",
    )
    parser.add_argument(
        "--scores", nargs="+", choices=sorted(SCORES), default=list(SCORES),
        help="Score columns to map (default: all)",
    )
    parser.add_argument(
        "--cell-size", type=float, default=DEFAULT_CELL_SIZE_M,
        help="Averaging grid cell size in metres; patches snap to this grid "
        f"(default: {DEFAULT_CELL_SIZE_M:g}, an eighth of a patch)",
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


def patch_squares(frame: pd.DataFrame) -> np.ndarray:
    import shapely

    half = PATCH_SIZE_M / 2
    x = frame["center_x"].to_numpy()
    y = frame["center_y"].to_numpy()
    return shapely.box(x - half, y - half, x + half, y + half)


def polygonal(geometries: np.ndarray) -> np.ndarray:
    """Keep only the polygon parts of each geometry (drops slivers of lines/points)."""
    import shapely

    geometries = geometries.copy()
    mixed = np.flatnonzero(~np.isin(shapely.get_type_id(geometries), (3, 6)))
    for index in mixed:
        parts = shapely.get_parts(geometries[index])
        parts = parts[np.isin(shapely.get_type_id(parts), (3, 6))]
        geometries[index] = shapely.union_all(parts) if len(parts) else shapely.Polygon()
    return geometries


def load_parks(path: Path, bounds: tuple[float, float, float, float]) -> np.ndarray:
    """Read the park polygons overlapping ``bounds`` (EPSG:5070) as valid geometries."""
    import geopandas as gpd
    import pyogrio
    import shapely
    from pyproj import Transformer

    if not path.is_file():
        raise FileNotFoundError(f"Missing parks file: {path}")
    source_crs = pyogrio.read_info(path)["crs"]
    if not source_crs:
        raise ValueError(f"Parks file has no CRS (missing .prj?): {path}")
    bbox = Transformer.from_crs(CRS, source_crs, always_xy=True).transform_bounds(
        *bounds, densify_pts=21,
    )
    parks = gpd.read_file(path, bbox=bbox, columns=[])
    geometries = shapely.make_valid(parks.to_crs(CRS).geometry.to_numpy())
    geometries = polygonal(geometries)
    geometries = geometries[~shapely.is_empty(geometries)]
    if len(geometries) == 0:
        raise ValueError(f"No park polygons in {path} overlap the patches")
    return geometries


def clip_to_parks(squares: np.ndarray, parks: np.ndarray) -> np.ndarray:
    """The part of each square inside any park (empty polygon when none)."""
    import shapely

    square_index, park_index = shapely.STRtree(parks).query(squares, predicate="intersects")
    pieces = polygonal(shapely.intersection(squares[square_index], parks[park_index]))
    clipped = np.full(len(squares), shapely.Polygon(), dtype=object)
    counts = np.bincount(square_index, minlength=len(squares))
    single = counts[square_index] == 1
    clipped[square_index[single]] = pieces[single]
    if (~single).any():
        grouped = pd.Series(pieces[~single]).groupby(square_index[~single])
        for index, group in grouped:
            clipped[index] = shapely.union_all(group.to_numpy())
    return clipped


class PatchGrid:
    """Regular grid over the patches; each patch covers a fixed block of cells.

    A patch's footprint snaps to the nearest cell edges, so with the default
    eighth-of-a-patch cell it covers exactly 8 x 8 cells, shifted by at most
    half a cell. Only the cells that some patch covers are stored.
    """

    def __init__(self, frame: pd.DataFrame, cell_size: float) -> None:
        if cell_size <= 0 or cell_size > PATCH_SIZE_M:
            raise ValueError(f"--cell-size must be in (0, {PATCH_SIZE_M:g}] metres")
        half = PATCH_SIZE_M / 2
        x = frame["center_x"].to_numpy()
        y = frame["center_y"].to_numpy()
        self.cell_size = cell_size
        n = max(1, round(PATCH_SIZE_M / cell_size))
        # Snap each footprint's top-left corner to multiples of the cell size.
        cols = np.rint((x - half) / cell_size).astype(np.int64)
        rows = np.rint(-(y + half) / cell_size).astype(np.int64)
        self.left = float(cols.min() * cell_size)
        self.top = float(-rows.min() * cell_size)
        cols -= cols.min()
        rows -= rows.min()
        self.shape = (int(rows.max()) + n, int(cols.max()) + n)
        offsets = np.arange(n)
        cells = (
            (rows[:, None, None] + offsets[None, :, None]) * self.shape[1]
            + cols[:, None, None] + offsets[None, None, :]
        ).reshape(len(x), n * n)
        cell_ids, inverse = np.unique(cells, return_inverse=True)
        self.patch_cells = inverse.reshape(cells.shape)
        self.rows, self.cols = np.divmod(cell_ids, self.shape[1])
        self.inside = np.ones(len(cell_ids), dtype=bool)

    @property
    def transform(self):
        from rasterio.transform import from_origin

        return from_origin(self.left, self.top, self.cell_size, self.cell_size)

    def mask_to(self, parks: np.ndarray) -> None:
        """Drop the cells whose centre is outside every park."""
        from rasterio.features import rasterize

        burned = rasterize(
            ((park, 1) for park in parks), out_shape=self.shape, transform=self.transform,
            fill=0, dtype="uint8",
        )
        self.inside = burned[self.rows, self.cols] == 1

    def _sum(self, keep: np.ndarray, values: np.ndarray | None) -> np.ndarray:
        cells = self.patch_cells[keep]
        weights = None if values is None else np.repeat(values[keep], cells.shape[1])
        return np.bincount(cells.ravel(), weights=weights, minlength=len(self.rows))

    def count(self) -> np.ndarray:
        """Patches covering each stored cell; NaN outside the parks."""
        count = self._sum(np.ones(len(self.patch_cells), dtype=bool), None)
        return np.where(self.inside, count, np.nan).astype(np.float32)

    def mean(self, values: np.ndarray) -> np.ndarray:
        """Mean of every patch covering each stored cell; NaN where none has a value."""
        keep = ~np.isnan(values)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = self._sum(keep, values) / self._sum(keep, None)
        return np.where(self.inside, mean, np.nan).astype(np.float32)

    def strips(self, cell_values: np.ndarray, height: int):
        """Yield (first row, dense block of ``height`` rows) down the grid."""
        # Cells are stored in row-major order, so each strip is a contiguous slice.
        bounds = np.searchsorted(self.rows, np.arange(0, self.shape[0] + height, height))
        for top, (start, stop) in enumerate(zip(bounds[:-1], bounds[1:])):
            first = top * height
            block = np.full(
                (min(height, self.shape[0] - first), self.shape[1]), np.nan, dtype=np.float32,
            )
            block[self.rows[start:stop] - first, self.cols[start:stop]] = cell_values[start:stop]
            yield first, block

    def display(self, cell_values: np.ndarray) -> tuple[np.ndarray, tuple[float, ...]]:
        """Cell values averaged into blocks of at most MAP_PIXELS per side, for PNGs.

        Averaging (rather than resampling) keeps small parks visible at state scale.
        """
        factor = max(1, math.ceil(max(self.shape) / MAP_PIXELS))
        shape = (math.ceil(self.shape[0] / factor), math.ceil(self.shape[1] / factor))
        valid = ~np.isnan(cell_values)
        blocks = (self.rows[valid] // factor) * shape[1] + self.cols[valid] // factor
        size = shape[0] * shape[1]
        with np.errstate(invalid="ignore", divide="ignore"):
            image = (
                np.bincount(blocks, weights=cell_values[valid], minlength=size)
                / np.bincount(blocks, minlength=size)
            )
        block = factor * self.cell_size
        extent = (
            self.left, self.left + shape[1] * block, self.top - shape[0] * block, self.top,
        )
        return image.reshape(shape), extent


def write_geotiff(path: Path, grid: PatchGrid, bands: dict[str, np.ndarray]) -> None:
    import rasterio
    from rasterio.windows import Window

    profile = {
        "driver": "GTiff", "height": grid.shape[0], "width": grid.shape[1],
        "count": len(bands), "dtype": "float32", "crs": CRS, "nodata": np.nan,
        "transform": grid.transform, "compress": "deflate", "BIGTIFF": "IF_SAFER",
    }
    if min(grid.shape) >= 256:
        profile.update(tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **profile) as dataset:
        for index, (name, cell_values) in enumerate(bands.items(), start=1):
            for first, block in grid.strips(cell_values, 256):
                window = Window(0, first, grid.shape[1], block.shape[0])
                dataset.write(block, index, window=window)
            dataset.set_band_description(index, name)


def draw_map(
    grid: PatchGrid, cell_values: np.ndarray, patches: int, score: str, state: str,
    path: Path, dpi: int, in_parks: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    title, vmin, vmax = SCORES[score]
    # One hue, light to dark; trimmed so the lowest values stay visible on white.
    cmap = LinearSegmentedColormap.from_list(
        "greens_trimmed", plt.get_cmap("Greens")(np.linspace(0.2, 1.0, 256)),
    ).with_extremes(bad=(1, 1, 1, 0))
    norm = Normalize(vmin=vmin, vmax=vmax)
    image, extent = grid.display(cell_values)

    fig, ax = plt.subplots(figsize=(8, 9), dpi=dpi)
    pad = 5_000
    ax.imshow(
        np.ma.masked_invalid(image), cmap=cmap, norm=norm, extent=extent,
        origin="upper", interpolation="nearest",
    )
    ax.set_xlim(extent[0] - pad, extent[1] + pad)
    ax.set_ylim(extent[2] - pad, extent[3] + pad)
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
        f"{patches:,} patches{' clipped to parks' if in_parks else ''} · "
        f"{PATCH_SIZE_M:g} m squares averaged on a {grid.cell_size:g} m grid · {CRS}",
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
    parks_path: Path | None = None,
) -> dict[str, int | str]:
    import geopandas as gpd
    import shapely

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
    squares = patch_squares(located)

    footprints = squares
    if parks_path is not None:
        parks = load_parks(parks_path, tuple(shapely.total_bounds(squares)))
        footprints = clip_to_parks(squares, parks)
        combined["park_fraction"] = np.nan
        combined.loc[located.index, "park_fraction"] = (
            shapely.area(footprints) / PATCH_SIZE_M**2
        ).round(4)
        located = combined.loc[located.index]
        grid.mask_to(parks)
    in_park = ~shapely.is_empty(footprints)

    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(combined_csv, index=False)
    if gpkg.exists():
        gpkg.unlink()
    patches = gpd.GeoDataFrame(
        located[in_park], geometry=shapely.force_2d(footprints[in_park]), crs=CRS,
    )
    patches.to_file(gpkg, layer=f"patches_USA_{state}", driver="GPKG", promote_to_multi=True)
    count = grid.count()
    means = {score: grid.mean(located[score].to_numpy(dtype=float)) for score in scores}
    write_geotiff(tif, grid, {**means, "patch_count": count})
    for score, png in zip(scores, pngs):
        draw_map(
            grid, means[score], int(in_park.sum()), score, state, png, dpi,
            parks_path is not None,
        )
    return {
        "state": state, "parts": len(parts), "rows": len(combined),
        "mapped": int(in_park.sum()), "outside_parks": int((~in_park).sum()),
        "missing_coordinates": missing, "max_overlap": int(np.nanmax(count, initial=0)),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    predictions_dir = Path(args.predictions_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve() if args.output_dir
        else predictions_dir / "maps"
    )
    parks_path = Path(args.parks).expanduser().resolve() if args.parks else None
    try:
        summary = map_state(
            predictions_dir, output_dir, args.scores, args.dpi, args.overwrite,
            args.cell_size, parks_path,
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
    if parks_path is not None:
        print(
            f"{summary['outside_parks']} patches lie entirely outside the parks and "
            "were left out of the GeoPackage and maps (kept in the combined CSV "
            "with park_fraction 0)"
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
