#!/usr/bin/env python3
"""Combine one state's coordinate-enriched prediction parts and map each score.

Reads every ``predictions_USA_XX_part_NNNNN_with_coordinates.csv`` in one state
folder, stacks them into a single table, turns each patch center into its
512 x 512 pixel footprint (0.6 m pixels, so 307.2 m squares in EPSG:5070), and
writes:

- ``combined_USA_XX.csv``      all rows from every part, in part order
- ``patch_scores_USA_XX.gpkg`` one square polygon per patch, every column kept
- ``map_USA_XX_<score>.png``   one static map per mapped score
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
        help="Destination for the combined CSV, GeoPackage and maps "
        "(default: <predictions-dir>/maps)",
    )
    parser.add_argument(
        "--scores", nargs="+", choices=sorted(SCORES), default=list(SCORES),
        help="Score columns to map (default: all)",
    )
    parser.add_argument(
        "--boundary",
        help="Optional vector file (e.g. state or county shapefile/GeoPackage) drawn "
        "as a grey outline under the patches; reprojected to EPSG:5070",
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


def load_boundary(path: Path):
    import geopandas as gpd

    boundary = gpd.read_file(path)
    if boundary.crs is None:
        raise ValueError(f"Boundary file has no CRS: {path}")
    return boundary.to_crs(CRS)


def draw_map(
    frame: pd.DataFrame, score: str, state: str, path: Path, dpi: int, boundary=None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    title, vmin, vmax = SCORES[score]
    # One hue, light to dark; trimmed so the lowest values stay visible on white.
    cmap = LinearSegmentedColormap.from_list(
        "greens_trimmed", plt.get_cmap("Greens")(np.linspace(0.2, 1.0, 256)),
    )
    norm = Normalize(vmin=vmin, vmax=vmax)
    data = frame.dropna(subset=[score]).sort_values(score)
    half = PATCH_SIZE_M / 2
    x = data["center_x"].to_numpy()[:, None]
    y = data["center_y"].to_numpy()[:, None]
    corners_x = x + np.array([-half, half, half, -half])
    corners_y = y + np.array([-half, -half, half, half])
    colors = cmap(norm(data[score].to_numpy()))
    # A thin same-colour edge keeps 307 m squares visible at state scale.
    squares = PolyCollection(
        np.stack([corners_x, corners_y], axis=-1),
        facecolors=colors, edgecolors=colors, linewidths=0.4,
    )

    fig, ax = plt.subplots(figsize=(8, 9), dpi=dpi)
    pad = 5_000
    xmin, xmax = frame["center_x"].min() - pad, frame["center_x"].max() + pad
    ymin, ymax = frame["center_y"].min() - pad, frame["center_y"].max() + pad
    if boundary is not None:
        boundary.boundary.plot(ax=ax, color="#bbbbbb", linewidth=0.6, zorder=1)
        bxmin, bymin, bxmax, bymax = boundary.total_bounds
        xmin, xmax = min(xmin, bxmin), max(xmax, bxmax)
        ymin, ymax = min(ymin, bymin), max(ymax, bymax)
    squares.set_zorder(2)
    ax.add_collection(squares)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(f"USA_{state} — {title}", loc="left", fontsize=13, color="#222222")
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
        orientation="horizontal", fraction=0.04, pad=0.03,
    )
    colorbar.set_label(score, color="#555555")
    colorbar.outline.set_visible(False)
    fig.text(
        0.01, 0.01,
        f"{len(data):,} patches · {PATCH_SIZE_M:g} m squares · {CRS}",
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
    boundary_path: Path | None = None,
) -> dict[str, int | str]:
    scores = scores or list(SCORES)
    state, parts = find_parts(predictions_dir)
    combined_csv = output_dir / f"combined_USA_{state}.csv"
    gpkg = output_dir / f"patch_scores_USA_{state}.gpkg"
    pngs = [output_dir / f"map_USA_{state}_{score}.png" for score in scores]
    if not overwrite:
        existing = [path for path in (combined_csv, gpkg, *pngs) if path.exists()]
        if existing:
            raise FileExistsError(f"Output already exists: {existing[0]}; use --overwrite")

    boundary = load_boundary(boundary_path) if boundary_path else None
    combined = combine_parts(parts, state)
    located = combined.dropna(subset=["center_x", "center_y"])
    missing = len(combined) - len(located)
    if located.empty:
        raise ValueError(f"No rows with coordinates for USA_{state}")

    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(combined_csv, index=False)
    if gpkg.exists():
        gpkg.unlink()
    patch_squares(located).to_file(gpkg, layer=f"patches_USA_{state}", driver="GPKG")
    for score, png in zip(scores, pngs):
        draw_map(located, score, state, png, dpi, boundary)
    return {
        "state": state, "parts": len(parts), "rows": len(combined),
        "mapped": len(located), "missing_coordinates": missing,
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
            Path(args.boundary).expanduser().resolve() if args.boundary else None,
        )
    except (OSError, ValueError) as exc:
        print(f"Mapping failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"USA_{summary['state']}: combined {summary['rows']} rows from "
        f"{summary['parts']} parts; mapped {summary['mapped']} patches "
        f"({len(args.scores)} maps) to {output_dir}"
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
