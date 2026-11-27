import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import face as facevalue
import facevalue as application


class FaceValueTests(unittest.TestCase):
    def test_no_arguments_do_nothing(self) -> None:
        with patch.object(facevalue, "supported_images") as supported_images:
            self.assertEqual(application.main([]), 0)
        supported_images.assert_not_called()

    def test_boolean_window_argument(self) -> None:
        self.assertTrue(application.parse_arguments(["--window", "true"]).window)
        self.assertFalse(application.parse_arguments(["--window", "false"]).window)

    def test_testmode_defaults_true_and_accepts_boolean(self) -> None:
        self.assertTrue(application.parse_arguments([]).testmode)
        self.assertFalse(
            application.parse_arguments(["--testmode", "false"]).testmode
        )

    def test_face_window_is_bounded_to_1024_by_768(self) -> None:
        data = {
            "images": [],
            "deepface": {"embeddings": []},
            "mediapipe": {"geometry": []},
            "consensus": {
                "status": "unavailable",
                "message": "No faces were recognized; consensus was not generated.",
            },
        }
        with patch("matplotlib.pyplot.show"):
            figure = facevalue.show_window(data, block=False)
        width_px, height_px = figure.get_size_inches() * figure.dpi

        self.assertLessEqual(width_px, 1024)
        self.assertLessEqual(height_px, 768)
        import matplotlib.pyplot as plt

        plt.close(figure)

    def test_axis_text_is_reduced_until_it_fits(self) -> None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(2, 2), dpi=100)
        canvas = FigureCanvasAgg(figure)
        axis = figure.add_subplot()
        title = axis.set_title("A title that is deliberately much too long", fontsize=24)

        facevalue.fit_axis_text(figure)

        renderer = canvas.get_renderer()
        title_width = title.get_window_extent(renderer=renderer).width
        axis_width = axis.get_window_extent(renderer=renderer).width
        self.assertLessEqual(title_width, axis_width * 0.95)
        self.assertLess(title.get_fontsize(), 24)

    def test_desktop_input_and_window_are_compatible(self) -> None:
        arguments = application.parse_arguments(
            [
                "--desktop",
                "desktop.json",
                "--input",
                "faces.json",
                "--window",
                "true",
            ]
        )

        self.assertEqual(arguments.desktop, Path("desktop.json"))
        self.assertEqual(arguments.input, Path("faces.json"))
        self.assertTrue(arguments.window)

    def test_input_argument_rejects_non_json_file(self) -> None:
        with self.assertRaises(SystemExit):
            application.parse_arguments(["--input", "data.txt"])

    def test_input_and_fdir_are_mutually_exclusive(self) -> None:
        with self.assertRaises(SystemExit):
            application.parse_arguments(
                ["--input", "data.json", "--fdir", "images"]
            )

    def test_run_loads_input_without_extracting_faces(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "input.json"
            data = {
                "images": [],
                "deepface": {"embeddings": []},
                "mediapipe": {"geometry": []},
                "consensus": {
                    "id": "consensus",
                    "status": "unavailable",
                    "source_image_ids": [],
                    "message": "No faces were recognized; consensus was not generated.",
                },
            }
            input_path.write_text(json.dumps(data), encoding="utf-8")
            arguments = argparse.Namespace(
                input=input_path, fdir=None, output=None, window=True
            )

            with (
                patch.object(facevalue, "show_window") as show_window,
                patch.object(facevalue, "supported_images") as supported_images,
            ):
                facevalue.run(
                    arguments.fdir,
                    getattr(arguments, "input", None),
                    arguments.output,
                    arguments.window,
                )

            supported_images.assert_not_called()
            show_window.assert_called_once_with(data)

    def test_display_only_image_does_not_change_consensus_sources(self) -> None:
        data = {
            "images": [],
            "deepface": {"embeddings": []},
            "mediapipe": {"geometry": []},
            "consensus": {
                "status": "ok",
                "source_image_ids": ["image-1"],
            },
        }
        geometry = {
            "image": {"path": "desktop.png", "width_px": 1, "height_px": 1}
        }
        embedding = {"embedding": [1.0]}

        with (
            patch.object(
                facevalue,
                "extract_mediapipe_geometry",
                return_value=geometry,
            ),
            patch.object(
                facevalue,
                "extract_deepface_embedding",
                return_value=embedding,
            ),
        ):
            facevalue.add_display_only_image(data, Path("desktop.png"))

        self.assertEqual(data["images"][-1]["id"], "desktop-face")
        self.assertEqual(data["mediapipe"]["geometry"][-1], geometry)
        self.assertEqual(data["deepface"]["embeddings"][-1], embedding)
        self.assertEqual(data["consensus"]["source_image_ids"], ["image-1"])

    def test_desktop_consensus_similarity_uses_cosine_score(self) -> None:
        data = {
            "deepface": {
                "embeddings": [
                    {
                        "image_id": "desktop-face",
                        "embedding": [1.0, 0.0],
                    }
                ]
            },
            "consensus": {
                "deepface": {
                    "embedding": [0.5, 0.5],
                }
            },
        }

        score = facevalue.desktop_consensus_similarity(data)

        self.assertAlmostEqual(score, 2 ** -0.5)

    def test_supported_images_uses_every_supported_file_in_name_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            (directory / "z.png").write_bytes(b"image")
            (directory / "middle.jpg").write_bytes(b"image")
            (directory / "notes.txt").write_text("not an image", encoding="utf-8")
            (directory / "a.JPEG").write_bytes(b"image")

            self.assertEqual(
                facevalue.supported_images(directory),
                [
                    directory / "a.JPEG",
                    directory / "middle.jpg",
                    directory / "z.png",
                ],
            )

    def test_procrustes_face_forward_aligns_landmarks(self) -> None:
        landmarks = [
            {"x": 0.0, "y": 0.0, "z": 0.0}
            for _ in range(max(facevalue.PROCRUSTES_ANCHOR_INDICES) + 1)
        ]
        for index, target in zip(
            facevalue.PROCRUSTES_ANCHOR_INDICES,
            facevalue.PROCRUSTES_TARGET,
            strict=True,
        ):
            landmarks[index] = {
                "x": target[0],
                "y": -target[2],
                "z": target[1],
            }

        result = facevalue.procrustes_face_forward(landmarks)

        self.assertEqual(result["landmark_count"], len(landmarks))
        self.assertEqual(len(result["embedding"]), len(landmarks) * 3)
        self.assertLess(result["rms_anchor_error"], 1e-10)

    def test_run_writes_expected_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            image_path_1 = directory / "face-1.png"
            image_path_2 = directory / "face-2.png"
            image_path_1.write_bytes(b"image")
            image_path_2.write_bytes(b"image")
            output_path = directory / "result.json"
            geometry_1 = {
                "landmark_count": 1,
                "landmarks": [{"x": 0.1, "y": 0.2, "z": 0.3}],
                "embedding": [0.1, 0.2, 0.3],
                "facial_transformation_matrixes": [],
                "face_forward": {
                    "method": "test",
                    "landmarks": [{"x": 0.1, "y": 0.2, "z": 0.3}],
                    "scale": 1.0,
                    "rms_anchor_error": 0.0,
                    "anchor_landmark_indices": [0],
                    "transform_matrix": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                },
                "image": {
                    "path": str(image_path_1),
                    "width_px": 10,
                    "height_px": 10,
                },
            }
            geometry_2 = {
                **geometry_1,
                "image": {**geometry_1["image"], "path": str(image_path_2)},
            }
            deepface_embedding_1 = {
                "model": "SFace",
                "dimension_count": 2,
                "embedding": [0.1, 0.2],
                "face_confidence": 0.99,
                "facial_area": {"x": 1, "y": 2, "w": 3, "h": 4},
                "image_path": str(image_path_1),
            }
            deepface_embedding_2 = {
                **deepface_embedding_1,
                "embedding": [0.3, 0.4],
                "image_path": str(image_path_2),
            }
            arguments = argparse.Namespace(
                fdir=directory, output=output_path, window=False
            )

            with (
                patch.object(
                    facevalue,
                    "extract_mediapipe_geometry",
                    side_effect=[geometry_1, geometry_2],
                ),
                patch.object(
                    facevalue,
                    "extract_deepface_embedding",
                    side_effect=[deepface_embedding_1, deepface_embedding_2],
                ),
            ):
                facevalue.run(
                    arguments.fdir,
                    getattr(arguments, "input", None),
                    arguments.output,
                    arguments.window,
                )

            data = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [item["id"] for item in data["images"]],
                ["image-1", "image-2"],
            )
            self.assertEqual(
                [item["image_id"] for item in data["deepface"]["embeddings"]],
                ["image-1", "image-2"],
            )
            self.assertEqual(
                [item["image_id"] for item in data["mediapipe"]["geometry"]],
                ["image-1", "image-2"],
            )
            self.assertEqual(len(data["deepface"]["embeddings"]), 2)
            self.assertEqual(len(data["mediapipe"]["geometry"]), 2)
            self.assertEqual(
                data["consensus"]["source_image_ids"],
                ["image-1", "image-2"],
            )
            self.assertEqual(
                data["consensus"]["deepface"]["dimension_count"], 2
            )
            self.assertEqual(
                data["consensus"]["mediapipe"]["geometry"]["landmark_count"], 1
            )
            self.assertEqual(
                data["consensus"]["mediapipe"]["embedding"]["dimension_count"], 3
            )
            self.assertEqual(
                len(data["consensus"]["mediapipe"]["source_transforms"]), 2
            )

    def test_run_records_failed_images_and_excludes_them_from_consensus(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            (directory / "face-1.png").write_bytes(b"image")
            (directory / "face-2.png").write_bytes(b"image")
            output_path = directory / "result.json"
            geometry = {
                "landmark_count": 1,
                "landmarks": [{"x": 0.1, "y": 0.2, "z": 0.3}],
                "embedding": [0.1, 0.2, 0.3],
                "facial_transformation_matrixes": [],
                "face_forward": {
                    "method": "test",
                    "landmarks": [{"x": 0.1, "y": 0.2, "z": 0.3}],
                    "scale": 1.0,
                    "rms_anchor_error": 0.0,
                    "anchor_landmark_indices": [0],
                    "transform_matrix": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                },
                "image": {
                    "path": str(directory / "face-1.png"),
                    "width_px": 10,
                    "height_px": 10,
                },
            }
            deepface_embedding = {
                "model": "SFace",
                "dimension_count": 2,
                "embedding": [0.1, 0.2],
                "face_confidence": 0.99,
                "facial_area": {"x": 1, "y": 2, "w": 3, "h": 4},
                "image_path": str(directory / "face-1.png"),
            }
            arguments = argparse.Namespace(
                fdir=directory, output=output_path, window=False
            )

            with (
                patch.object(
                    facevalue,
                    "extract_mediapipe_geometry",
                    side_effect=[
                        facevalue.FaceDetectionError("no face detected"),
                        geometry,
                    ],
                ),
                patch.object(
                    facevalue,
                    "extract_deepface_embedding",
                    return_value=deepface_embedding,
                ),
            ):
                facevalue.run(
                    arguments.fdir,
                    getattr(arguments, "input", None),
                    arguments.output,
                    arguments.window,
                )

            data = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(data["images"][0]["status"], "failed")
            self.assertTrue(data["images"][0]["path"].endswith("face-1.png"))
            self.assertEqual(data["consensus"]["source_image_ids"], ["image-2"])
            self.assertEqual(len(data["deepface"]["embeddings"]), 1)
            self.assertEqual(len(data["mediapipe"]["geometry"]), 1)

    def test_no_detected_faces_skip_consensus(self) -> None:
        image_results = [
            {
                "id": "image-1",
                "path": Path("missing-face.png"),
                "status": "failed",
                "error": "no face detected",
            }
        ]

        data = facevalue.build_output(image_results)

        self.assertEqual(data["consensus"]["status"], "unavailable")
        self.assertEqual(data["consensus"]["source_image_ids"], [])
        self.assertIn("No faces were recognized", data["consensus"]["message"])
        self.assertEqual(data["images"][0]["path"], "missing-face.png")


if __name__ == "__main__":
    unittest.main()
