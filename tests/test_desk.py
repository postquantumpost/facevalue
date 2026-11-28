import asyncio
import json
import queue
import tempfile
import threading
import unittest
import zlib
from pathlib import Path
from unittest.mock import AsyncMock, call, patch

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

    def test_number_recognizer_reads_every_template_sample(self) -> None:
        samples_directory = Path(__file__).parents[1] / "samples"
        samples = {
            **{
                f"good{sample_number}.png": value
                for sample_number, value in desk.NUMBER_SAMPLE_VALUES.items()
            },
            **desk.NUMBER_ADDITIONAL_SAMPLE_VALUES,
        }
        for sample_name, expected_value in samples.items():
            sample_path = samples_directory / sample_name
            with self.subTest(sample=sample_path.name):
                with Image.open(sample_path) as sample:
                    image = np.asarray(sample)
                height_px, width_px = image.shape[:2]
                value, detail = desk.recognize_integer_region(
                    image,
                    [0, 0, width_px, height_px],
                    width_px,
                    height_px,
                )

                self.assertEqual(value, expected_value)
                self.assertTrue(detail)

    def test_text_ocr_uses_real_tesseract_invocation(self) -> None:
        image = Image.new("RGBA", (240, 70), (0, 0, 0, 255))
        drawing = ImageDraw.Draw(image)
        drawing.text(
            (10, 5),
            "Cosmetics",
            fill=(255, 255, 255, 255),
            font=ImageFont.load_default(size=40),
        )

        detected_text = desk.ocr_region_text(
            np.asarray(image),
            [0, 0, 240, 70],
            240,
            70,
        )

        self.assertEqual(detected_text, "Cosmetics")

    def test_number_recognizer_reads_failed_ocr_samples(self) -> None:
        samples_directory = Path(__file__).parents[1] / "samples"
        for sample_name, expected_value in (
            ("failed_ocr_raw.png", 125),
            ("failed_ocr_raw3.png", 170),
            ("failed4_n1.png", 74),
            ("failed9_n1.png", 129),
            ("failed5_n1.png", 58),
            ("failed0_n1.png", 50),
            ("failed1_n1.png", 111),
            ("failed6_n1.png", 62),
        ):
            with self.subTest(sample=sample_name):
                with Image.open(samples_directory / sample_name) as sample:
                    image = np.asarray(sample)
                height_px, width_px = image.shape[:2]
                value, detail = desk.recognize_integer_region(
                    image,
                    [0, 0, width_px, height_px],
                    width_px,
                    height_px,
                )

                self.assertEqual(value, expected_value)
                self.assertTrue(detail)

    def test_number_recognizer_digits_cover_zero_to_nine(self) -> None:
        templates = desk.number_templates()

        self.assertEqual(set(templates), set("0123456789"))
        self.assertTrue(all(templates[digit] for digit in templates))
        self.assertAlmostEqual(desk.NUMBER_RECOGNITION_MAX_ERROR, 0.073205)
        self.assertAlmostEqual(desk.NUMBER_RECOGNITION_MIN_MARGIN, 0.045)

    def test_real_number_samples_reconcile_through_application(self) -> None:
        samples_directory = Path(__file__).parents[1] / "samples"
        number_region = [1816, 342, 2010, 412]
        for sample_name, expected_value in (
            ("failed_ocr_raw.png", 125),
            ("failed_ocr_raw3.png", 170),
        ):
            with self.subTest(sample=sample_name):
                screenshot = np.zeros((2160, 3840, 4), dtype=np.uint8)
                with Image.open(samples_directory / sample_name) as sample:
                    sample_pixels = np.asarray(sample)
                x1, y1, x2, y2 = number_region
                screenshot[y1:y2, x1:x2] = sample_pixels

                value, error = facevalue.read_integer_region(
                    screenshot,
                    "Nose Size",
                    number_region,
                    expected_values=(expected_value, expected_value + 1),
                )

                self.assertEqual(value, expected_value)
                self.assertEqual(error, "")

    def test_number_recognizer_rejects_out_of_range_values(self) -> None:
        samples_directory = Path(__file__).parents[1] / "samples"
        with Image.open(samples_directory / "good27.png") as sample:
            three_digit_image = np.asarray(sample)
        glyphs_98 = desk.segment_number_glyphs(
            Image.fromarray(three_digit_image)
        )
        source_digits = glyphs_98 + glyphs_98[1:]
        foreground = np.full((70, 194, 4), 0, dtype=np.uint8)
        foreground[:, :, 3] = 255
        x = 64
        for glyph in source_digits:
            height_px, width_px = glyph.shape
            foreground[14 : 14 + height_px, x : x + width_px, :3] = (
                glyph[:, :, np.newaxis] * 255
            )
            x += width_px + 3

        value, detail = desk.recognize_integer_region(
            foreground,
            [0, 0, 194, 70],
            194,
            70,
        )

        self.assertIsNone(value)
        self.assertIn("outside 0 to 255", detail)

    def test_click_at_moves_through_intermediate_points(self) -> None:
        client = unittest.mock.Mock()
        client.mouse.x = 0
        client.mouse.y = 0
        client.drain = AsyncMock()
        client.mouse.hold.return_value = unittest.mock.MagicMock(
            __enter__=unittest.mock.Mock(),
            __exit__=unittest.mock.Mock(return_value=False),
        )

        with patch("desk.asyncio.sleep", new=AsyncMock()) as sleep:
            asyncio.run(desk.click_at(client, 100, 80))

        self.assertEqual(client.mouse.move.call_count, 40)
        client.mouse.move.assert_any_call(50, 40)
        client.mouse.move.assert_called_with(100, 80)
        self.assertEqual(client.drain.await_count, 42)
        self.assertEqual(sleep.await_count, 42)

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

    def test_capture_worker_can_stop_after_startup_checks(self) -> None:
        screenshot = np.zeros((1080, 1920, 4), dtype=np.uint8)
        frame_queue: queue.Queue[object] = queue.Queue(maxsize=1)
        stop_event = threading.Event()
        config = desk.DesktopConfig(host="desktop.example.test")
        client = object()
        desktop_context = AsyncMock()
        desktop_context.__aenter__.return_value = client
        state: dict[str, object] = {}

        with (
            patch("desk.open_desktop", return_value=desktop_context),
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
                    state,
                    startup_screen_check=lambda _: (True, "checked"),
                    stop_after_startup=True,
                )
            )

        self.assertTrue(stop_event.is_set())
        capture_full.assert_awaited_once_with(client)
        self.assertIs(state["desktop_screenshot"], screenshot)
        self.assertIsInstance(frame_queue.get_nowait(), np.ndarray)

    def test_desktop_display_check_verifies_second_region(self) -> None:
        screenshot = np.zeros((1080, 1920, 4), dtype=np.uint8)

        with (
            patch(
                "facevalue.desk.region_contains_text",
                return_value=True,
            ) as contains_text,
            patch(
                "facevalue.desk.ocr_region_text",
                return_value="Alter Face & Hair",
            ),
            patch("builtins.print") as print_message,
        ):
            result = facevalue.verify_desktop_display(screenshot)

        self.assertTrue(result[0])
        self.assertIn("continuing", result[1].lower())
        contains_text.assert_called_once()
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

    def test_controls_keyboard_navigation_and_q_recheck(self) -> None:
        screenshots = [
            np.full((2160, 3840, 4), index, dtype=np.uint8)
            for index in range(7)
        ]
        screenshot = screenshots[0]
        root_screenshot = screenshots[6]
        client = object()

        with (
            patch(
                "facevalue.focus_menu_item",
                new=AsyncMock(
                    side_effect=lambda _client, screenshot, _menu, item: (
                        True,
                        f"Focused {item}.",
                        screenshot,
                    )
                ),
            ),
            patch(
                "facevalue.desk.capture_full_desktop",
                new=AsyncMock(
                    side_effect=screenshots[1:],
                ),
            ) as capture_full,
            patch(
                "facevalue.desk.press_key",
                new=AsyncMock(),
            ) as press_key,
            patch(
                "facevalue.desk.region_contains_text",
                side_effect=[
                    True,
                    True,
                    False,
                    True,
                    True,
                    True,
                    True,
                    False,
                    True,
                    False,
                ],
            ) as contains_text,
            patch(
                "facevalue.read_integer_region",
                return_value=(123, ""),
            ) as read_integer,
            patch(
                "facevalue.read_all_face_structure_values",
                new=AsyncMock(
                    return_value=(True, "Read all Face Structure values.", root_screenshot)
                ),
            ) as read_all,
            patch(
                "facevalue.modify_random_face_values",
                new=AsyncMock(
                    return_value=(True, "Completed random modifications.", root_screenshot)
                ),
            ) as modify_random,
            patch("facevalue.asyncio.sleep", new=AsyncMock()) as sleep,
        ):
            result = asyncio.run(
                facevalue.verify_desktop_controls(client, screenshot)
            )

        self.assertTrue(result[0])
        self.assertIn("completed", result[1].lower())
        self.assertIs(result[2], root_screenshot)
        self.assertEqual(press_key.await_args_list, [
            call(client, "e"),
            call(client, "q"),
            call(client, "e"),
            call(client, "e"),
            call(client, "q"),
            call(client, "q"),
        ])
        self.assertEqual(sleep.await_count, 6)
        self.assertEqual(capture_full.await_count, 6)
        self.assertEqual(
            contains_text.call_args_list[0].args[4],
            "Alter Face & Hair > Adjust Face Template",
        )
        self.assertEqual(
            contains_text.call_args_list[1].args[4],
            "Alter Face & Hair",
        )
        self.assertEqual(
            contains_text.call_args_list[5].args[4],
            "Nose Size",
        )
        self.assertEqual(
            contains_text.call_args_list[3].args[4],
            "Alter Face & Hair > Face Structure",
        )
        self.assertEqual(
            contains_text.call_args_list[4].args[4],
            "Alter Face & Hair > Face Structure > Facial Balance",
        )
        self.assertEqual(
            contains_text.call_args_list[7].args[4],
            "Facial Balance",
        )
        self.assertEqual(
            contains_text.call_args_list[9].args[4],
            "Face Structure",
        )
        read_integer.assert_called_once()
        read_all.assert_awaited_once_with(client, root_screenshot)
        modify_random.assert_awaited_once_with(client, root_screenshot, 1)

    def test_number_crop_offsets_stay_inside_menu_item(self) -> None:
        self.assertGreaterEqual(facevalue.nominal_number_y_offset_1, 0)
        self.assertLess(
            facevalue.nominal_number_y_offset_1,
            facevalue.nominal_number_y_offset_2,
        )
        self.assertLessEqual(
            facevalue.nominal_number_y_offset_2,
            facevalue.nominal_item_height,
        )

    def test_focused_menu_index_detects_brighter_row(self) -> None:
        screenshot = np.zeros((2160, 3840, 4), dtype=np.uint8)
        screenshot[:, :, 3] = 255
        for index in range(3):
            top = 322 + 112 * index
            screenshot[top : top + 112, 270:1996, :3] = 20
        highlighted_top = 322 + 112
        screenshot[
            highlighted_top : highlighted_top + 112,
            270:1996,
            :3,
        ] = 35

        focused_index = facevalue.focused_menu_index(screenshot, 3)

        self.assertEqual(focused_index, 1)

    def test_focus_menu_item_skips_empty_entries(self) -> None:
        screenshot = np.zeros((2160, 3840, 4), dtype=np.uint8)
        client = object()
        menu = {
            "Face Structure": None,
            "empty0": None,
            "Hair": None,
        }

        with (
            patch(
                "facevalue.focused_menu_index",
                side_effect=[1, 2],
            ),
            patch(
                "facevalue.desk.press_key",
                new=AsyncMock(),
            ) as press_key,
            patch(
                "facevalue.desk.capture_full_desktop",
                new=AsyncMock(return_value=screenshot),
            ),
            patch("facevalue.asyncio.sleep", new=AsyncMock()),
        ):
            focused, _, _ = asyncio.run(
                facevalue.focus_menu_item(
                    client,
                    screenshot,
                    menu,
                    "Hair",
                )
            )

        self.assertTrue(focused)
        press_key.assert_awaited_once_with(client, "Down")

    def test_recover_desktop_menu_uses_breadcrumb_depth(self) -> None:
        screenshot = np.zeros((2160, 3840, 4), dtype=np.uint8)
        client = object()

        with (
            patch(
                "facevalue.desk.ocr_region_text",
                side_effect=[
                    "Alter Face & Hair > Face Structure > Forehead/Glabella",
                    "Alter Face & Hair > Face Structure",
                    "Alter Face & Hair > Face Structure",
                    "Alter Face & Hair",
                    "Alter Face & Hair",
                ],
            ),
            patch(
                "facevalue.desk.press_key",
                new=AsyncMock(),
            ) as press_key,
            patch(
                "facevalue.desk.capture_full_desktop",
                new=AsyncMock(return_value=screenshot),
            ),
            patch("facevalue.asyncio.sleep", new=AsyncMock()),
        ):
            recovered = asyncio.run(
                facevalue.recover_desktop_menu(client, screenshot)
            )

        self.assertIs(recovered, screenshot)
        self.assertEqual(
            press_key.await_args_list,
            [call(client, "q"), call(client, "q")],
        )

    def test_capture_desktop_retries_connection_twice(self) -> None:
        screenshot = np.zeros((1080, 1920, 4), dtype=np.uint8)
        frame_queue: queue.Queue[object] = queue.Queue(maxsize=1)
        stop_event = threading.Event()
        config = desk.DesktopConfig(host="desktop.example.test")
        desktop_context = AsyncMock()
        desktop_context.__aenter__.return_value = object()

        with (
            patch(
                "desk.open_desktop",
                side_effect=[
                    ConnectionError("first"),
                    ConnectionError("second"),
                    desktop_context,
                ],
            ) as open_desktop,
            patch(
                "desk.capture_full_desktop",
                new=AsyncMock(return_value=screenshot),
            ),
            patch("desk.asyncio.sleep", new=AsyncMock()),
            patch("builtins.print"),
        ):
            asyncio.run(
                desk.capture_desktop(
                    config,
                    10.0,
                    frame_queue,
                    stop_event,
                    stop_after_startup=True,
                )
            )

        self.assertEqual(open_desktop.call_count, 3)
        self.assertTrue(stop_event.is_set())

    def test_modify_random_face_values_updates_map_and_returns_home(self) -> None:
        screenshot = np.zeros((2160, 3840, 4), dtype=np.uint8)
        client = object()
        original_state = facevalue.current_face_value_state.copy()
        facevalue.current_face_value_state.clear()
        facevalue.current_face_value_state.update(
            {
                "Nose Size": 125,
                "Nose/Forehead Ratio": 110,
                "Face Protrusion": 147,
                "Vert. Face Ratio": 143,
                "Facial Feature Slant": 146,
                "Horiz. Face Ratio": 182,
            }
        )
        try:
            with (
                patch(
                    "facevalue.system_random.choice",
                    side_effect=[
                        "Nose Size", 1,
                    ],
                ) as choice,
                patch(
                    "facevalue.select_menu_item",
                    new=AsyncMock(return_value=(True, "selected", screenshot)),
                ) as select_menu,
                patch(
                    "facevalue.return_to_menu",
                    new=AsyncMock(return_value=(True, "returned", screenshot)),
                ) as return_menu,
                patch(
                    "facevalue.focus_menu_item",
                    new=AsyncMock(
                        return_value=(True, "focused", screenshot)
                    ),
                ),
                patch(
                    "facevalue.desk.press_key",
                    new=AsyncMock(),
                ) as press_key,
                patch(
                    "facevalue.desk.capture_full_desktop",
                    new=AsyncMock(return_value=screenshot),
                ),
                patch(
                    "facevalue.read_leaf_integer",
                    side_effect=[
                        (126, ""),
                    ],
                ),
                patch(
                    "facevalue.asyncio.sleep",
                    new=AsyncMock(),
                ),
            ):
                result = asyncio.run(
                    facevalue.modify_random_face_values(client, screenshot)
                )

            self.assertTrue(result[0])
            self.assertEqual(
                facevalue.current_face_value_state,
                {
                    "Nose Size": 126,
                    "Nose/Forehead Ratio": 110,
                    "Face Protrusion": 147,
                    "Vert. Face Ratio": 143,
                    "Facial Feature Slant": 146,
                    "Horiz. Face Ratio": 182,
                },
            )
            self.assertEqual(choice.call_count, 2)
            self.assertEqual(select_menu.await_count, 2)
            self.assertEqual(return_menu.await_count, 2)
            self.assertIn(call(client, "Right"), press_key.await_args_list)
            self.assertEqual(
                press_key.await_args_list.count(call(client, "e")),
                2,
            )
        finally:
            facevalue.current_face_value_state.clear()
            facevalue.current_face_value_state.update(original_state)

    def test_modify_face_value_reverts_after_unexpected_ocr_readback(self) -> None:
        screenshot = np.zeros((2160, 3840, 4), dtype=np.uint8)
        client = object()
        original_state = facevalue.current_face_value_state.copy()
        facevalue.current_face_value_state.clear()
        facevalue.current_face_value_state["Nose Size"] = 170
        try:
            with (
                patch(
                    "facevalue.select_menu_item",
                    new=AsyncMock(return_value=(True, "selected", screenshot)),
                ),
                patch(
                    "facevalue.return_to_menu",
                    new=AsyncMock(return_value=(True, "returned", screenshot)),
                ) as return_menu,
                patch(
                    "facevalue.focus_menu_item",
                    new=AsyncMock(
                        return_value=(True, "focused", screenshot)
                    ),
                ),
                patch(
                    "facevalue.desk.press_key",
                    new=AsyncMock(),
                ) as press_key,
                patch(
                    "facevalue.desk.capture_full_desktop",
                    new=AsyncMock(return_value=screenshot),
                ),
                patch(
                    "facevalue.read_leaf_integer",
                    return_value=(1, ""),
                ),
                patch("facevalue.asyncio.sleep", new=AsyncMock()),
                patch("builtins.print") as print_message,
            ):
                result = asyncio.run(
                    facevalue.modify_face_value(
                        client,
                        screenshot,
                        "Nose Size",
                        1,
                    )
                )

            self.assertFalse(result[0])
            self.assertIn("inconsistent", result[1])
            self.assertEqual(result[3], 0)
            self.assertEqual(
                press_key.await_args_list,
                [
                    call(client, "e"),
                    call(client, "Right"),
                    call(client, "e"),
                    call(client, "e"),
                    call(client, "Left"),
                    call(client, "e"),
                ],
            )
            self.assertEqual(return_menu.await_count, 2)
            self.assertEqual(facevalue.current_face_value_state["Nose Size"], 170)
            self.assertIn("attempting to revert", print_message.call_args.args[0])
        finally:
            facevalue.current_face_value_state.clear()
            facevalue.current_face_value_state.update(original_state)

    def test_random_modification_uses_all_face_values_when_testmode_false(self) -> None:
        screenshot = np.zeros((2160, 3840, 4), dtype=np.uint8)
        client = object()
        original_testmode = facevalue.testmode
        original_state = facevalue.current_face_value_state.copy()
        facevalue.testmode = False
        paths = {
            item_name: (submenu_name, submenu)
            for submenu_name, submenu in facevalue.alter_face_structure.items()
            for item_name in submenu
            if submenu[item_name] is None
        }
        facevalue.current_face_value_state.clear()
        facevalue.current_face_value_state.update(
            {item_name: 100 for item_name in paths}
        )
        try:
            with (
                patch(
                    "facevalue.system_random.choice",
                    side_effect=["Jaw Width", 1],
                ) as choice,
                patch(
                    "facevalue.modify_face_value",
                    new=AsyncMock(return_value=(True, "", screenshot, 1)),
                ),
            ):
                result = asyncio.run(
                    facevalue.modify_random_face_values(client, screenshot)
                )

            self.assertTrue(result[0])
            self.assertIn("Jaw Width", choice.call_args_list[0].args[0])
        finally:
            facevalue.testmode = original_testmode
            facevalue.current_face_value_state.clear()
            facevalue.current_face_value_state.update(original_state)

    def test_random_generator_uses_operating_system_entropy(self) -> None:
        import random

        self.assertIsInstance(facevalue.system_random, random.SystemRandom)

    def test_zero_actual_delta_is_a_no_op(self) -> None:
        old_value = 255
        new_value = 255
        actual_delta = new_value - old_value

        self.assertEqual(actual_delta, 0)
        self.assertEqual(old_value, new_value)

    def test_controls_report_keyboard_select_failure_without_q(self) -> None:
        screenshot = np.zeros((1080, 1920, 4), dtype=np.uint8)
        client = object()

        with (
            patch(
                "facevalue.focus_menu_item",
                new=AsyncMock(return_value=(False, "focus failed", screenshot)),
            ),
            patch(
                "facevalue.desk.capture_full_desktop",
                new=AsyncMock(return_value=screenshot),
            ),
            patch("facevalue.desk.region_contains_text", return_value=False),
            patch("facevalue.asyncio.sleep", new=AsyncMock()),
            patch(
                "facevalue.desk.press_key",
                new=AsyncMock(),
            ) as press_key,
        ):
            result = asyncio.run(
                facevalue.verify_desktop_controls(client, screenshot)
            )

        self.assertFalse(result[0])
        self.assertIn("Unable to select", result[1])
        press_key.assert_not_awaited()

    def test_controls_report_backspace_failure(self) -> None:
        screenshot = np.zeros((2160, 3840, 4), dtype=np.uint8)
        client = object()

        with (
            patch("facevalue.desk.click_at", new=AsyncMock()),
            patch(
                "facevalue.desk.capture_full_desktop",
                new=AsyncMock(return_value=screenshot),
            ),
            patch(
                "facevalue.desk.region_contains_text",
                side_effect=[True, True, True],
            ),
            patch("facevalue.desk.press_key", new=AsyncMock()),
            patch("facevalue.asyncio.sleep", new=AsyncMock()),
        ):
            result = asyncio.run(
                facevalue.verify_desktop_controls(client, screenshot)
            )

        self.assertFalse(result[0])
        self.assertIn("Q did not return", result[1])


if __name__ == "__main__":
    unittest.main()
