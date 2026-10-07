from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts import map_state_scores


def prediction_rows(start: int, count: int, state: str = "AL") -> pd.DataFrame:
    rows = []
    for index in range(start, start + count):
        row = {
            "image_filename": f"img_{index}.jpg",
            "state_code": state,
            "park_code": "04033-1402",
            "image_relative_path": f"USA_{state}_x/park/jpg/img_{index}.jpg",
            "center_x": 978336.0 + 400 * index,
            "center_y": 1119477.0 + 400 * index,
        }
        for score, (_, low, high) in map_state_scores.SCORES.items():
            row[score] = low + (high - low) * (index % 5) / 4
        rows.append(row)
    return pd.DataFrame(rows)


class MapStateScoresTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.state_dir = Path(self.temporary.name) / "USA_AL"
        self.state_dir.mkdir()
        self.output_dir = self.state_dir / "maps"

    def write_part(self, number: int, frame: pd.DataFrame, state: str = "AL") -> None:
        name = f"predictions_USA_{state}_part_{number:05d}_with_coordinates.csv"
        frame.to_csv(self.state_dir / name, index=False)

    def test_combines_parts_in_order_and_writes_outputs(self) -> None:
        import geopandas as gpd

        self.write_part(2, prediction_rows(3, 2))
        self.write_part(1, prediction_rows(0, 3))
        # Raw parts without coordinates must be ignored.
        prediction_rows(0, 1).to_csv(self.state_dir / "predictions_USA_AL_part_00001.csv")

        summary = map_state_scores.map_state(self.state_dir, self.output_dir, dpi=40)

        self.assertEqual(summary["parts"], 2)
        self.assertEqual(summary["rows"], 5)
        combined = pd.read_csv(self.output_dir / "combined_USA_AL.csv")
        self.assertEqual(list(combined["image_filename"]), [f"img_{i}.jpg" for i in range(5)])
        patches = gpd.read_file(self.output_dir / "patch_scores_USA_AL.gpkg")
        self.assertEqual(patches.crs.to_epsg(), 5070)
        self.assertEqual(len(patches), 5)
        bounds = patches.geometry.iloc[0].bounds
        self.assertAlmostEqual(bounds[2] - bounds[0], 307.2)
        self.assertAlmostEqual(bounds[3] - bounds[1], 307.2)
        self.assertAlmostEqual((bounds[0] + bounds[2]) / 2, 978336.0)
        for score in map_state_scores.SCORES:
            self.assertTrue((self.output_dir / f"map_USA_AL_{score}.png").is_file())
        self.assertTrue((self.output_dir / "mean_scores_USA_AL.tif").is_file())

    def test_overlapping_patches_are_averaged(self) -> None:
        import rasterio

        frame = prediction_rows(0, 2)
        # The second patch sits half a patch east, so the two share one column of cells.
        frame.loc[1, ["center_x", "center_y"]] = [
            frame.loc[0, "center_x"] + map_state_scores.PATCH_SIZE_M / 2,
            frame.loc[0, "center_y"],
        ]
        frame["score_ev"] = [1.0, 5.0]
        self.write_part(1, frame)

        summary = map_state_scores.map_state(
            self.state_dir, self.output_dir, scores=["score_ev"], dpi=40, cell_size=153.6,
        )

        self.assertEqual(summary["max_overlap"], 2)
        with rasterio.open(self.output_dir / "mean_scores_USA_AL.tif") as dataset:
            self.assertEqual(dataset.crs.to_epsg(), 5070)
            self.assertEqual(dataset.descriptions, ("score_ev", "patch_count"))
            self.assertEqual((dataset.height, dataset.width), (2, 3))
            self.assertAlmostEqual(dataset.res[0], 153.6)
            mean, count = dataset.read(1), dataset.read(2)
        for row in range(2):
            self.assertEqual(list(mean[row]), [1.0, 3.0, 5.0])
            self.assertEqual(list(count[row]), [1.0, 2.0, 1.0])

    def test_rows_without_coordinates_are_kept_in_csv_but_not_mapped(self) -> None:
        frame = prediction_rows(0, 3)
        frame[["center_x", "center_y"]] = frame[["center_x", "center_y"]].astype(object)
        frame.loc[1, ["center_x", "center_y"]] = ""
        self.write_part(1, frame)

        summary = map_state_scores.map_state(
            self.state_dir, self.output_dir, scores=["score_ev"], dpi=40,
        )

        self.assertEqual((summary["rows"], summary["mapped"]), (3, 2))
        self.assertEqual(summary["missing_coordinates"], 1)

    def test_parks_clip_patches_and_mask_cells(self) -> None:
        import geopandas as gpd
        import rasterio
        import shapely

        frame = prediction_rows(0, 3)
        self.write_part(1, frame)
        x0, y0 = frame.loc[0, "center_x"], frame.loc[0, "center_y"]
        half = map_state_scores.PATCH_SIZE_M / 2
        # One park covering the west half of patch 0; patches 1 and 2 are outside it.
        park = shapely.box(x0 - half - 50, y0 - half - 50, x0, y0 + half + 50)
        parks_path = Path(self.temporary.name) / "parks.gpkg"
        gpd.GeoDataFrame({"name": ["park"]}, geometry=[park], crs="EPSG:5070").to_crs(
            "EPSG:4326"
        ).to_file(parks_path)

        summary = map_state_scores.map_state(
            self.state_dir, self.output_dir, scores=["score_ev"], dpi=40,
            cell_size=153.6, parks_path=parks_path,
        )

        self.assertEqual((summary["mapped"], summary["outside_parks"]), (1, 2))
        combined = pd.read_csv(self.output_dir / "combined_USA_AL.csv")
        self.assertAlmostEqual(combined.loc[0, "park_fraction"], 0.5, places=3)
        self.assertEqual(list(combined.loc[1:, "park_fraction"]), [0.0, 0.0])
        patches = gpd.read_file(self.output_dir / "patch_scores_USA_AL.gpkg")
        self.assertEqual(list(patches["image_filename"]), ["img_0.jpg"])
        self.assertAlmostEqual(
            patches.geometry.iloc[0].area, map_state_scores.PATCH_SIZE_M**2 / 2, delta=50,
        )
        with rasterio.open(self.output_dir / "mean_scores_USA_AL.tif") as dataset:
            count = dataset.read(2)
        # Only patch 0's west column of 153.6 m cells (2 cells) lies inside the park.
        self.assertEqual(int((count > 0).sum()), 2)

    def test_refuses_duplicates_across_parts(self) -> None:
        self.write_part(1, prediction_rows(0, 2))
        self.write_part(2, prediction_rows(1, 2))
        with self.assertRaisesRegex(ValueError, "Duplicate prediction"):
            map_state_scores.map_state(self.state_dir, self.output_dir, dpi=40)

    def test_protects_existing_outputs(self) -> None:
        self.write_part(1, prediction_rows(0, 2))
        map_state_scores.map_state(self.state_dir, self.output_dir, ["veg_ev"], dpi=40)
        with self.assertRaises(FileExistsError):
            map_state_scores.map_state(self.state_dir, self.output_dir, ["veg_ev"], dpi=40)
        map_state_scores.map_state(
            self.state_dir, self.output_dir, ["veg_ev"], dpi=40, overwrite=True,
        )

    def test_requires_enriched_parts(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "append_patch_coordinates"):
            map_state_scores.map_state(self.state_dir, self.output_dir)


if __name__ == "__main__":
    unittest.main()
