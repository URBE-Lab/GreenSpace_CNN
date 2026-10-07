from __future__ import annotations

import csv
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import map_all_states, map_state_scores
from tests.test_map_state_scores import prediction_rows


class MapAllStatesTests(unittest.TestCase):
    def setUp(self) -> None:
        import geopandas as gpd
        import shapely

        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.states_dir = root / "run" / "states"
        self.output_dir = root / "run" / "maps_USA"
        # Two states 100 km apart, three patches each.
        self.offsets = {"AL": 0.0, "GA": 100_000.0}
        for state, offset in self.offsets.items():
            frame = prediction_rows(0, 3, state)
            frame["center_x"] += offset
            self.write_part(state, frame)
        # One park around every patch.
        x = prediction_rows(0, 3)["center_x"]
        y = prediction_rows(0, 3)["center_y"]
        parks = [
            shapely.box(x.min() - 1000 + offset, y.min() - 1000, x.max() + 1000 + offset,
                        y.max() + 1000)
            for offset in self.offsets.values()
        ]
        self.parks = root / "parks.gpkg"
        gpd.GeoDataFrame(geometry=parks, crs="EPSG:5070").to_file(self.parks)

    def write_part(self, state: str, frame: pd.DataFrame) -> None:
        folder = self.states_dir / f"USA_{state}"
        folder.mkdir(parents=True, exist_ok=True)
        frame.to_csv(
            folder / f"predictions_USA_{state}_part_00001_with_coordinates.csv", index=False,
        )

    def run_main(self, *extra: str) -> int:
        argv = [
            "--states-dir", str(self.states_dir), "--parks", str(self.parks),
            "--cell-size", "153.6", "--dpi", "40", *extra,
        ]
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            return map_all_states.main(argv)

    def test_maps_every_state_and_combines_them(self) -> None:
        import geopandas as gpd
        import rasterio

        self.assertEqual(self.run_main(), 0)

        combined = pd.read_csv(self.output_dir / "combined_USA.csv")
        self.assertEqual(list(combined["state_code"]), ["AL"] * 3 + ["GA"] * 3)
        self.assertIn("park_fraction", combined.columns)
        patches = gpd.read_file(self.output_dir / "patch_scores_USA.gpkg")
        self.assertEqual(len(patches), 6)
        self.assertEqual(patches.crs.to_epsg(), 5070)
        for score in map_state_scores.SCORES:
            self.assertTrue((self.output_dir / f"map_USA_{score}.png").is_file())

        al = rasterio.open(self.states_dir / "USA_AL" / "maps" / "mean_scores_USA_AL.tif")
        ga = rasterio.open(self.states_dir / "USA_GA" / "maps" / "mean_scores_USA_GA.tif")
        with al, ga, rasterio.open(self.output_dir / "mean_scores_USA.vrt") as mosaic:
            self.assertEqual(mosaic.crs.to_epsg(), 5070)
            self.assertEqual(mosaic.descriptions, al.descriptions)
            self.assertEqual(mosaic.res, al.res)
            for state in (al, ga):
                window = rasterio.windows.from_bounds(*state.bounds, mosaic.transform)
                np.testing.assert_array_equal(
                    mosaic.read(1, window=window.round_offsets().round_lengths()),
                    state.read(1),
                )

        with (self.output_dir / "state_status_USA.csv").open(newline="") as handle:
            statuses = {row["state"]: row["status"] for row in csv.DictReader(handle)}
        self.assertEqual(statuses, {"AL": "mapped", "GA": "mapped"})

    def test_reuses_mapped_states_and_records_failures(self) -> None:
        self.assertEqual(self.run_main(), 0)
        (self.states_dir / "USA_TX").mkdir()

        self.assertEqual(self.run_main(), 1)

        with (self.output_dir / "state_status_USA.csv").open(newline="") as handle:
            rows = {row["state"]: row for row in csv.DictReader(handle)}
        self.assertEqual(rows["AL"]["status"], "existing")
        self.assertEqual(rows["TX"]["status"], "failed")
        self.assertIn("--image-root", rows["TX"]["message"])
        # The national outputs are still rebuilt from the states that worked.
        combined = pd.read_csv(self.output_dir / "combined_USA.csv")
        self.assertEqual(len(combined), 6)


if __name__ == "__main__":
    unittest.main()
