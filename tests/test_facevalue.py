import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import facevalue


class FaceValueTests(unittest.TestCase):
    def test_no_arguments_do_nothing(self) -> None:
        with patch.object(facevalue, "first_image") as first_image:
            self.assertEqual(facevalue.main([]), 0)
        first_image.assert_not_called()

    def test_boolean_window_argument(self) -> None:
        self.assertTrue(facevalue.parse_arguments(["--window", "true"]).window)
        self.assertFalse(facevalue.parse_arguments(["--window", "false"]).window)

    def test_first_image_uses_supported_files_in_name_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            (directory / "z.png").write_bytes(b"image")
            (directory / "notes.txt").write_text("not an image", encoding="utf-8")
            (directory / "a.JPEG").write_bytes(b"image")

            self.assertEqual(facevalue.first_image(directory), directory / "a.JPEG")

    def test_run_writes_expected_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            image_path = directory / "face.png"
            image_path.write_bytes(b"image")
            output_path = directory / "result.json"
            geometry = {
                "landmark_count": 1,
                "landmarks": [{"x": 0.1, "y": 0.2, "z": 0.3}],
                "embedding": [0.1, 0.2, 0.3],
                "facial_transformation_matrixes": [],
                "image": {
                    "path": str(image_path),
                    "width_px": 10,
                    "height_px": 10,
                },
            }
            arguments = argparse.Namespace(
                fdir=directory, output=output_path, window=False
            )

            with patch.object(
                facevalue, "extract_mediapipe_geometry", return_value=geometry
            ):
                facevalue.run(arguments)

            data = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(data["deepface"], {"embeddings": []})
            self.assertEqual(data["mediapipe"]["geometry"], [geometry])


if __name__ == "__main__":
    unittest.main()
