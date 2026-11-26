import asyncio
import json
import queue
import tempfile
import threading
import unittest
import zlib
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

    def test_tight_compression_stream_survives_rectangle_boundaries(self) -> None:
        compressor = zlib.compressobj()
        first_pixels = b"\x00" * 12
        second_pixels = b"\xff" * 12
        first_compressed = compressor.compress(first_pixels)
        first_compressed += compressor.flush(zlib.Z_SYNC_FLUSH)
        second_compressed = compressor.compress(second_pixels)
        second_compressed += compressor.flush(zlib.Z_SYNC_FLUSH)
        streams: dict[int, object] = {}

        class Reader:
            def __init__(self, payload: bytes) -> None:
                self.payload = payload
                self.offset = 0

            async def readexactly(self, size: int) -> bytes:
                result = self.payload[self.offset : self.offset + size]
                self.offset += size
                return result

        class Client:
            def __init__(self, payload: bytes) -> None:
                self.reader = Reader(payload)

        async def decode_part(payload: bytes) -> np.ndarray:
            target = np.zeros((1, 6, 4), dtype=np.uint8)
            control_and_length = bytes((0x00, len(payload)))
            client = Client(control_and_length + payload)
            await desk.read_tight_rectangle(
                client,
                target,
                0,
                0,
                6,
                1,
                streams,
            )
            return target

        first = asyncio.run(decode_part(first_compressed))
        second = asyncio.run(decode_part(second_compressed))

        self.assertTrue((first[..., 3] == 255).all())
        self.assertTrue((second[..., :3] == 255).all())
        self.assertIn(0, streams)

    def test_failed_startup_check_stops_capture(self) -> None:
        screenshot = np.zeros((1080, 1920, 4), dtype=np.uint8)
        frame_queue: queue.Queue[object] = queue.Queue(maxsize=1)
        stop_event = threading.Event()
        config = desk.DesktopConfig(host="desktop.example.test")
        client = object()
        desktop_context = AsyncMock()
        desktop_context.__aenter__.return_value = client

        with (
            patch("desk.open_desktop", return_value=desktop_context),
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

    def test_capture_worker_reuses_one_desktop_connection(self) -> None:
        screenshot = np.zeros((1080, 1920, 4), dtype=np.uint8)
        updated_screenshot = np.ones((1080, 1920, 4), dtype=np.uint8)
        frame_queue: queue.Queue[object] = queue.Queue(maxsize=1)
        stop_event = threading.Event()
        config = desk.DesktopConfig(host="desktop.example.test")
        client = object()
        desktop_context = AsyncMock()
        desktop_context.__aenter__.return_value = client

        async def verify_control(received_client: object, _: object):
            self.assertIs(received_client, client)
            stop_event.set()
            return True, "Control verified.", updated_screenshot

        with (
            patch(
                "desk.open_desktop",
                return_value=desktop_context,
            ) as open_desktop,
            patch(
                "desk.capture_full_desktop",
                new=AsyncMock(return_value=screenshot),
            ) as capture_full,
            patch("builtins.print"),
        ):
            asyncio.run(
                desk.capture_desktop(
                    config,
                    10.0,
                    frame_queue,
                    stop_event,
                    startup_control_check=verify_control,
                )
            )

        open_desktop.assert_called_once_with(config)
        capture_full.assert_awaited_once_with(client)

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

    def test_mouse_control_clicks_adjust_face_template_and_rechecks(self) -> None:
        screenshot = np.zeros((2160, 3840, 4), dtype=np.uint8)
        updated_screenshot = np.ones((2160, 3840, 4), dtype=np.uint8)
        client = object()

        with (
            patch(
                "facevalue.desk.click_at",
                new=AsyncMock(),
            ) as click_at,
            patch(
                "facevalue.desk.capture_full_desktop",
                new=AsyncMock(return_value=updated_screenshot),
            ),
            patch(
                "facevalue.desk.region_contains_text",
                return_value=True,
            ) as contains_text,
            patch("facevalue.asyncio.sleep", new=AsyncMock()) as sleep,
        ):
            result = asyncio.run(
                facevalue.verify_mouse_control(client, screenshot)
            )

        self.assertTrue(result[0])
        self.assertIn("continuing", result[1])
        self.assertIs(result[2], updated_screenshot)
        click_at.assert_awaited_once_with(client, 310, 189)
        sleep.assert_awaited_once_with(1.0)
        self.assertEqual(
            contains_text.call_args.args[4],
            "Alter Face & Hair > Adjust Face Template",
        )

    def test_mouse_control_reports_click_failure(self) -> None:
        screenshot = np.zeros((1080, 1920, 4), dtype=np.uint8)
        client = object()

        with (
            patch("facevalue.desk.click_at", new=AsyncMock()),
            patch(
                "facevalue.desk.capture_full_desktop",
                new=AsyncMock(return_value=screenshot),
            ),
            patch("facevalue.desk.region_contains_text", return_value=False),
            patch("facevalue.asyncio.sleep", new=AsyncMock()),
        ):
            result = asyncio.run(
                facevalue.verify_mouse_control(client, screenshot)
            )

        self.assertFalse(result[0])
        self.assertIn("Unable to click", result[1])


if __name__ == "__main__":
    unittest.main()
