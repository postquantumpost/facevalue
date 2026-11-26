import asyncio
import json
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import AsyncMock, patch

import numpy as np

import desk
import facevalue


class DesktopTests(unittest.TestCase):
    def test_read_config_with_required_and_optional_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "desktop.json"
            config_path.write_text(
                json.dumps(
                    {
                        "host": "desktop.example.test",
                        "port": 5901,
                        "username": "operator",
                        "password": "secret",
                    }
                ),
                encoding="utf-8",
            )

            config = desk.read_config(config_path)

            self.assertEqual(config.host, "desktop.example.test")
            self.assertEqual(config.port, 5901)
            self.assertEqual(config.username, "operator")
            self.assertEqual(config.password, "secret")

    def test_scale_desktop_preserves_aspect_ratio_in_bounded_canvas(self) -> None:
        image = np.zeros((100, 200, 4), dtype=np.uint8)
        image[:, :, 3] = 255

        scaled = desk.scale_desktop(image)

        self.assertEqual(scaled.shape, (480, 640, 4))
        self.assertFalse(scaled[:80, :, :3].any())

    def test_region_contains_text_uses_normalized_ocr_output(self) -> None:
        image = np.zeros((1080, 1920, 4), dtype=np.uint8)
        image[:, :, 3] = 255
        result = CompletedProcess(
            args=[],
            returncode=0,
            stdout=b"  CoSmeTics \n",
            stderr=b"",
        )

        with patch("desk.subprocess.run", return_value=result):
            contains_text = desk.region_contains_text(
                image,
                [231, 82, 507, 145],
                3840,
                2160,
                "Cosmetics",
            )

        self.assertTrue(contains_text)

    def test_failed_startup_check_stops_capture(self) -> None:
        screenshot = np.zeros((1080, 1920, 4), dtype=np.uint8)
        frame_queue: queue.Queue[object] = queue.Queue(maxsize=1)
        stop_event = threading.Event()
        config = desk.DesktopConfig(host="desktop.example.test")

        with (
            patch("desk.capture_full_desktop", new=AsyncMock(return_value=screenshot)),
            patch("builtins.print"),
        ):
            asyncio.run(
                desk.capture_desktop(
                    config,
                    10.0,
                    frame_queue,
                    stop_event,
                    startup_screen_check=lambda _: (False, "Cosmetics is not displayed; exiting."),
                )
            )

        self.assertTrue(stop_event.is_set())
        self.assertIsInstance(frame_queue.get_nowait(), RuntimeError)

    def test_desktop_display_check_verifies_second_region(self) -> None:
        screenshot = np.zeros((1080, 1920, 4), dtype=np.uint8)

        with (
            patch(
                "facevalue.desk.region_contains_text",
                side_effect=[True, True],
            ) as contains_text,
            patch("builtins.print") as print_message,
        ):
            result = facevalue.verify_desktop_display(screenshot)

        self.assertTrue(result[0])
        self.assertIn("continuing", result[1])
        self.assertEqual(contains_text.call_count, 2)
        self.assertEqual(
            contains_text.call_args_list[1].args[1],
            facevalue.nominal_alter_face_and_hair,
        )
        self.assertEqual(
            contains_text.call_args_list[1].args[4],
            "Alter Face & Hair",
        )
        print_message.assert_called_once_with(
            "Verified Cosmetics is displayed.",
            flush=True,
        )

    def test_desktop_display_check_stops_after_missing_cosmetics(self) -> None:
        screenshot = np.zeros((1080, 1920, 4), dtype=np.uint8)

        with (
            patch(
                "facevalue.desk.region_contains_text",
                return_value=False,
            ) as contains_text,
            patch("builtins.print") as print_message,
        ):
            result = facevalue.verify_desktop_display(screenshot)

        self.assertFalse(result[0])
        self.assertIn("Cosmetics is not displayed", result[1])
        contains_text.assert_called_once()
        print_message.assert_not_called()


if __name__ == "__main__":
    unittest.main()
