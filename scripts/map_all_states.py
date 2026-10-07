#!/usr/bin/env python3
"""Map every state of an inference run, then combine them into national outputs.

For each ``USA_XX`` folder under ``--states-dir`` this runs
``map_state_scores.py`` (and, with ``--image-root``, the coordinate join first
when a state has no ``*_with_coordinates.csv`` parts yet). A state whose maps
already exist is reused unless ``--overwrite`` is given; a state that fails is
recorded and the run continues. The national outputs are then rebuilt from
every state folder with complete maps:

- ``combined_USA.csv``         every state's combined CSV, stacked
- ``patch_scores_USA.gpkg``    every state's patch polygons in one layer
- ``mean_scores_USA.vrt``      a mosaic of the state GeoTIFFs (open in QGIS/ArcGIS)
- ``map_USA_<score>.png``      one national map per score (lower 48 and DC)
- ``state_status_USA.csv``     what happened to each state in this run
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import append_patch_coordinates, map_state_scores as mss  # noqa: E402


STATE_DIR = re.compile(r"^USA_([A-Z]{2})$")
# Mapped in their own state maps but left off the national PNGs, which show the
# lower 48 and DC; the GeoPackage, CSV and VRT still include them.
NOT_CONTIGUOUS = {"AK", "HI", "PR", "VI", "GU", "AS", "MP"}
STRIP_ROWS = 256
STATUS_COLUMNS = [
    "state", "status", "rows", "mapped", "outside_parks", "missing_coordinates",
    "seconds", "message",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--states-dir", required=True,
        help="The run's states folder containing USA_XX prediction folders",
    )
    parser.add_argument(
        "--output-dir",
        help="Destination for the national outputs (default: <states-dir>/../maps_USA)",
    )
    parser.add_argument(
        "--parks",
        help="Park polygon file (e.g. ParkServe_Parks.shp) passed to every state",
    )
    parser.add_argument(
        "--image-root",
        help="Inference image root; when given, states without *_with_coordinates.csv "
        "parts get the coordinate join first",
    )
    parser.add_argument(
        "--skip-bad-tables", action="store_true",
        help="Passed to the coordinate join (see append_patch_coordinates.py)",
    )
    parser.add_argument(
        "--states", nargs="+", metavar="XX",
        help="Only map these state codes (default: every USA_XX folder); the national "
        "outputs still combine every state with complete maps",
    )
    parser.add_argument(
        "--scores", nargs="+", choices=sorted(mss.SCORES), default=list(mss.SCORES),
        help="Score columns to map (default: all)",
    )
    parser.add_argument(
        "--cell-size", type=float, default=mss.DEFAULT_CELL_SIZE_M,
        help=f"Averaging grid cell size in metres (default: {mss.DEFAULT_CELL_SIZE_M:g}); "
        "must be the same for every state",
    )
    parser.add_argument("--dpi", type=int, default=200, help="PNG resolution (default: 200)")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Remap states that already have maps (national outputs are always rebuilt)",
    )
    return parser


def find_states(states_dir: Path) -> dict[str, Path]:
    if not states_dir.is_dir():
        raise FileNotFoundError(f"Missing states directory: {states_dir}")
    states = {
        m.group(1): path for path in sorted(states_dir.iterdir())
        if path.is_dir() and (m := STATE_DIR.fullmatch(path.name))
    }
    if not states:
        raise FileNotFoundError(f"No USA_XX folders in {states_dir}")
    return states


def state_outputs(state_dir: Path, state: str, scores: list[str]) -> dict[str, Path]:
    maps = state_dir / "maps"
    return {
        "csv": maps / f"combined_USA_{state}.csv",
        "gpkg": maps / f"patch_scores_USA_{state}.gpkg",
        "tif": maps / f"mean_scores_USA_{state}.tif",
        **{score: maps / f"map_USA_{state}_{score}.png" for score in scores},
    }


def has_enriched_parts(state_dir: Path) -> bool:
    return any(mss.PART_NAME.fullmatch(path.name) for path in state_dir.iterdir())


def map_one_state(state: str, state_dir: Path, args: argparse.Namespace) -> dict:
    outputs = state_outputs(state_dir, state, args.scores)
    if not args.overwrite and all(path.exists() for path in outputs.values()):
        return {"status": "existing"}
    status = "mapped"
    if not has_enriched_parts(state_dir):
        if not args.image_root:
            raise FileNotFoundError(
                "no *_with_coordinates.csv parts; pass --image-root to run the "
                "coordinate join, or run append_patch_coordinates.py first"
            )
        append_patch_coordinates.enrich_parts(
            Path(args.image_root).expanduser().resolve(), state_dir, state_dir,
            skip_bad_tables=args.skip_bad_tables,
        )
        status = "joined and mapped"
    # Partial outputs here come from an interrupted run of this script, so replace them.
    summary = mss.map_state(
        state_dir, state_dir / "maps", args.scores, args.dpi, overwrite=True,
        cell_size=args.cell_size,
        parks_path=Path(args.parks).expanduser().resolve() if args.parks else None,
    )
    return {"status": status, **summary}


def csv_header(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle))


def union(column_lists: list[list[str]]) -> list[str]:
    columns: list[str] = []
    for names in column_lists:
        columns += [name for name in names if name not in columns]
    return columns


def combine_csvs(paths: list[Path], target: Path) -> None:
    import pandas as pd

    headers = [csv_header(path) for path in paths]
    columns = union(headers)
    with target.open("w", encoding="utf-8", newline="") as out:
        csv.writer(out).writerow(columns)
        for path, header in zip(paths, headers):
            if header == columns:
                # Same columns: copy the rows verbatim, skipping the header line.
                with path.open(encoding="utf-8-sig", newline="") as source:
                    source.readline()
                    shutil.copyfileobj(source, out, 1 << 20)
                continue
            for chunk in pd.read_csv(
                path, dtype=str, keep_default_na=False, chunksize=200_000,
                encoding="utf-8-sig",
            ):
                chunk.reindex(columns=columns, fill_value="").to_csv(
                    out, header=False, index=False, lineterminator="\n",
                )


def combine_geopackages(paths: list[Path], target: Path) -> int:
    import pyogrio

    columns = union([list(pyogrio.read_info(path)["fields"]) for path in paths])
    target.unlink(missing_ok=True)
    features = 0
    for index, path in enumerate(paths):
        frame = pyogrio.read_dataframe(path)
        frame = frame.reindex(columns=[*columns, "geometry"])
        pyogrio.write_dataframe(
            frame, target, layer="patches_USA", driver="GPKG", append=index > 0,
            promote_to_multi=True,
        )
        features += len(frame)
    return features


def read_grids(paths: dict[str, Path]) -> dict[str, dict]:
    """Origin, size and band names of each state GeoTIFF, checked to share one grid."""
    import rasterio

    grids = {}
    for state, path in paths.items():
        with rasterio.open(path) as dataset:
            transform = dataset.transform
            grids[state] = {
                "path": path, "left": transform.c, "top": transform.f,
                "cell": transform.a, "width": dataset.width, "height": dataset.height,
                "bands": list(dataset.descriptions),
            }
    cells = {round(grid["cell"], 6) for grid in grids.values()}
    if len(cells) != 1:
        raise ValueError(
            f"State GeoTIFFs use different cell sizes ({sorted(cells)}); remap the odd "
            "states with --overwrite --states XX and the same --cell-size"
        )
    return grids


def place(grid: dict, left: float, top: float, cell: float) -> tuple[int, int]:
    """Row and column offset of a state grid inside a grid starting at left, top."""
    return round((top - grid["top"]) / cell), round((grid["left"] - left) / cell)


def write_vrt(target: Path, grids: dict[str, dict], bands: list[str]) -> None:
    from rasterio.crs import CRS

    cell = next(iter(grids.values()))["cell"]
    left = min(grid["left"] for grid in grids.values())
    top = max(grid["top"] for grid in grids.values())
    right = max(grid["left"] + grid["width"] * cell for grid in grids.values())
    bottom = min(grid["top"] - grid["height"] * cell for grid in grids.values())
    width, height = round((right - left) / cell), round((top - bottom) / cell)
    lines = [
        f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">',
        f'  <SRS dataAxisToSRSAxisMapping="1,2">{escape(CRS.from_string(mss.CRS).to_wkt())}</SRS>',
        f"  <GeoTransform>{left!r}, {cell!r}, 0, {top!r}, 0, {-cell!r}</GeoTransform>",
    ]
    for number, band in enumerate(bands, start=1):
        lines += [
            f'  <VRTRasterBand dataType="Float32" band="{number}">',
            f"    <Description>{escape(band)}</Description>",
            "    <NoDataValue>nan</NoDataValue>",
        ]
        for grid in grids.values():
            row, col = place(grid, left, top, cell)
            try:
                source = Path(os.path.relpath(grid["path"], target.parent)).as_posix()
                relative = 1
            except ValueError:  # different Windows drives
                source, relative = str(grid["path"]), 0
            lines += [
                "    <ComplexSource>",
                f'      <SourceFilename relativeToVRT="{relative}">{escape(source)}</SourceFilename>',
                f"      <SourceBand>{grid['bands'].index(band) + 1}</SourceBand>",
                f'      <SrcRect xOff="0" yOff="0" xSize="{grid["width"]}" ySize="{grid["height"]}"/>',
                f'      <DstRect xOff="{col}" yOff="{row}" xSize="{grid["width"]}" ySize="{grid["height"]}"/>',
                "      <NODATA>nan</NODATA>",
                "    </ComplexSource>",
            ]
        lines.append("  </VRTRasterBand>")
    lines.append("</VRTDataset>")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def national_images(
    grids: dict[str, dict], scores: list[str],
) -> tuple[dict[str, np.ndarray], tuple[float, ...]]:
    """Block-average every contiguous state's GeoTIFF into one national image per score."""
    import rasterio
    from rasterio.windows import Window

    contiguous = {state: grid for state, grid in grids.items() if state not in NOT_CONTIGUOUS}
    if not contiguous:
        raise ValueError("No contiguous-US states to draw on the national maps")
    cell = next(iter(contiguous.values()))["cell"]
    left = min(grid["left"] for grid in contiguous.values())
    top = max(grid["top"] for grid in contiguous.values())
    right = max(grid["left"] + grid["width"] * cell for grid in contiguous.values())
    bottom = min(grid["top"] - grid["height"] * cell for grid in contiguous.values())
    shape = (round((top - bottom) / cell), round((right - left) / cell))
    blocks = {score: mss.BlockAverage(left, top, cell, shape) for score in scores}
    for grid in contiguous.values():
        row_offset, col_offset = place(grid, left, top, cell)
        indexes = [grid["bands"].index(score) + 1 for score in scores]
        with rasterio.open(grid["path"]) as dataset:
            for first in range(0, grid["height"], STRIP_ROWS):
                rows = min(STRIP_ROWS, grid["height"] - first)
                strip = dataset.read(indexes, window=Window(0, first, grid["width"], rows))
                for score, values in zip(scores, strip):
                    r, c = np.nonzero(~np.isnan(values))
                    blocks[score].add(r + first + row_offset, c + col_offset, values[r, c])
    extent = next(iter(blocks.values())).extent
    return {score: block.image() for score, block in blocks.items()}, extent


def build_national(
    states: dict[str, Path], scores: list[str], output_dir: Path, dpi: int,
) -> dict[str, int]:
    import pyogrio

    complete = {
        state: state_outputs(path, state, scores)
        for state, path in states.items()
        if all(p.exists() for p in state_outputs(path, state, scores).values())
    }
    if not complete:
        raise ValueError("No state has complete maps to combine")
    clipped = {
        state: "park_fraction" in csv_header(paths["csv"]) for state, paths in complete.items()
    }
    if len(set(clipped.values())) > 1:
        unclipped = sorted(state for state, value in clipped.items() if not value)
        raise ValueError(
            "Some states were mapped without --parks (" + ", ".join(unclipped)
            + "); remap them with --overwrite --states ... --parks"
        )
    grids = read_grids({state: paths["tif"] for state, paths in complete.items()})
    bands = [b for b in [*scores, "patch_count"] if all(b in g["bands"] for g in grids.values())]
    missing = sorted(set(scores) - set(bands))
    if missing:
        print(f"WARNING: not every state has {', '.join(missing)}; left off the national maps")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Combining {len(complete)} states into {output_dir}", flush=True)
    combine_csvs([paths["csv"] for paths in complete.values()], output_dir / "combined_USA.csv")
    patches = combine_geopackages(
        [paths["gpkg"] for paths in complete.values()], output_dir / "patch_scores_USA.gpkg",
    )
    write_vrt(output_dir / "mean_scores_USA.vrt", grids, bands)
    mapped_scores = [band for band in bands if band in scores]
    images, extent = national_images(grids, mapped_scores)
    contiguous_patches = sum(
        pyogrio.read_info(paths["gpkg"])["features"]
        for state, paths in complete.items() if state not in NOT_CONTIGUOUS
    )
    footnote = mss.map_footnote(
        contiguous_patches, next(iter(clipped.values())), next(iter(grids.values()))["cell"],
    )
    for score, image in images.items():
        mss.draw_map(
            image, extent, score, "USA (lower 48 and DC)", footnote,
            output_dir / f"map_USA_{score}.png", dpi,
        )
    return {"states": len(complete), "patches": patches}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    states_dir = Path(args.states_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve() if args.output_dir
        else states_dir.parent / "maps_USA"
    )
    try:
        states = find_states(states_dir)
    except OSError as exc:
        print(f"Mapping failed: {exc}", file=sys.stderr)
        return 1
    selected = [code.upper() for code in args.states] if args.states else list(states)
    unknown = sorted(set(selected) - set(states))
    if unknown:
        print(f"Mapping failed: no USA_XX folder for {', '.join(unknown)}", file=sys.stderr)
        return 1

    statuses = []
    for number, state in enumerate(selected, start=1):
        print(f"[{number}/{len(selected)}] USA_{state} ...", flush=True)
        started = time.time()
        try:
            result = map_one_state(state, states[state], args)
            message = ""
        except Exception as exc:  # keep going; one bad state must not stop the run
            result = {"status": "failed"}
            message = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        result.update(state=state, seconds=round(time.time() - started), message=message)
        statuses.append(result)
        detail = f" ({result['mapped']} patches mapped)" if "mapped" in result else ""
        print(f"  {result['status']}{detail}{': ' + message if message else ''}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "state_status_USA.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(statuses)
    failed = [item["state"] for item in statuses if item["status"] == "failed"]

    try:
        national = build_national(states, args.scores, output_dir, args.dpi)
    except (OSError, ValueError) as exc:
        print(f"National combine failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"National outputs: {national['states']} states, {national['patches']} patches "
        f"in {output_dir}"
    )
    if failed:
        print(
            f"WARNING: {len(failed)} states failed ({', '.join(failed)}); see "
            f"state_status_USA.csv. Fix them and rerun with --states {' '.join(failed)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
